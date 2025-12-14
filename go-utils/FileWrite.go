package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

type WriteResult struct {
	Success      bool   `json:"success"`
	BytesWritten int    `json:"bytes_written"`
	Message      string `json:"message"`
	Error        string `json:"error,omitempty"`
}

func main() {
	pathPtr := flag.String("path", "", "Full path of the file to write")
	contentPtr := flag.String("content", "", "Content to write to the file")
	appendPtr := flag.Bool("append", false, "Append to file instead of overwriting")
	flag.Parse()

	if *pathPtr == "" {
		printResult(false, 0, "", "File path is required")
		return
	}

	dir := filepath.Dir(*pathPtr)
	if err := os.MkdirAll(dir, 0755); err != nil {
		printResult(false, 0, "", fmt.Sprintf("Failed to create directory structure: %v", err))
		return
	}

	flags := os.O_CREATE | os.O_WRONLY
	if *appendPtr {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}

	f, err := os.OpenFile(*pathPtr, flags, 0644)
	if err != nil {
		printResult(false, 0, "", fmt.Sprintf("Failed to open file: %v", err))
		return
	}
	defer f.Close()

	n, err := f.WriteString(*contentPtr)
	if err != nil {
		printResult(false, 0, "", fmt.Sprintf("Failed to write content: %v", err))
		return
	}

	mode := "created/overwritten"
	if *appendPtr {
		mode = "appended"
	}

	printResult(true, n, fmt.Sprintf("Successfully %s file.", mode), "")
}

func printResult(success bool, bytes int, msg string, errStr string) {
	res := WriteResult{
		Success:      success,
		BytesWritten: bytes,
		Message:      msg,
		Error:        errStr,
	}
	data, _ := json.Marshal(res)
	fmt.Println(string(data))
}
