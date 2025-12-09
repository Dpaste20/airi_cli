package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type SearchResult struct {
	Files []string `json:"files"`
	Count int      `json:"count"`
	Error string   `json:"error,omitempty"`
}

var skipDirs = map[string]bool{
	"/proc": true, "/sys": true, "/dev": true, "/run": true,
	"/tmp": true, "/var/run": true, "/snap": true,
}

func main() {
	queryPtr := flag.String("query", "", "Search query (filename substring)")
	limitPtr := flag.Int("limit", 20, "Max results to return")
	rootPtr := flag.String("root", "/", "Root directory to start search")
	timeoutPtr := flag.Int("timeout", 10, "Timeout in seconds")
	flag.Parse()

	if *queryPtr == "" {
		printResult([]string{}, "Query cannot be empty")
		return
	}

	query := strings.ToLower(*queryPtr)
	results := []string{}
	limit := *limitPtr

	done := make(chan bool)

	go func() {
		filepath.WalkDir(*rootPtr, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				if os.IsPermission(err) {
					return nil
				}
				return nil
			}

			if d.IsDir() {
				if skipDirs[path] {
					return filepath.SkipDir
				}
				if strings.HasPrefix(d.Name(), ".") && len(d.Name()) > 1 {
					return filepath.SkipDir
				}
			}

			if !d.IsDir() {
				if strings.Contains(strings.ToLower(d.Name()), query) {
					results = append(results, path)
					if len(results) >= limit {
						return fmt.Errorf("limit_reached")
					}
				}
			}
			return nil
		})
		done <- true
	}()

	select {
	case <-done:
		printResult(results, "")
	case <-time.After(time.Duration(*timeoutPtr) * time.Second):
		printResult(results, "Search timed out (partial results returned)")
	}
}

func printResult(files []string, err string) {
	res := SearchResult{
		Files: files,
		Count: len(files),
		Error: err,
	}
	data, _ := json.Marshal(res)
	fmt.Println(string(data))
}
