package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

type LogResponse struct {
	LinesReturned int      `json:"lines_returned,omitempty"`
	Logs          []string `json:"logs,omitempty"`
	Error         string   `json:"error,omitempty"`
}

func main() {
	linesPtr := flag.Int("lines", 100, "Number of lines to return")
	filePtr := flag.String("file", "/var/log/syslog", "Path to the log file")
	flag.Parse()

	file, err := os.Open(*filePtr)
	if err != nil {
		if os.IsPermission(err) {
			printError("Permission denied. Try running the agent with sudo.")
		} else if os.IsNotExist(err) {
			printError("System log file not found or inaccessible.")
		} else {
			printError(err.Error())
		}
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var buffer []string
	limit := *linesPtr

	for scanner.Scan() {
		buffer = append(buffer, scanner.Text())
		if len(buffer) > limit {
			buffer = buffer[1:]
		}
	}

	if err := scanner.Err(); err != nil {
		printError(fmt.Sprintf("Error reading file stream: %v", err))
		return
	}

	resp := LogResponse{
		LinesReturned: len(buffer),
		Logs:          buffer,
	}

	output, _ := json.Marshal(resp)
	fmt.Println(string(output))
}

func printError(msg string) {
	resp := LogResponse{Error: msg}
	output, _ := json.Marshal(resp)
	fmt.Println(string(output))
}
