package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
)

type ModifyResult struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	Error   string `json:"error,omitempty"`
}

func main() {
	pathPtr := flag.String("path", "", "Full path of the file to modify")
	oldTextPtr := flag.String("old", "", "Text to search for")
	newTextPtr := flag.String("new", "", "Text to replace it with")
	flag.Parse()

	if *pathPtr == "" || *oldTextPtr == "" {
		printResult(false, "File path and old text are required", "")
		return
	}

	content, err := os.ReadFile(*pathPtr)
	if err != nil {
		printResult(false, "", fmt.Sprintf("Failed to read file: %v", err))
		return
	}

	fileText := string(content)

	if !strings.Contains(fileText, *oldTextPtr) {
		printResult(false, "", "Target text not found in file. Modification aborted.")
		return
	}

	newContent := strings.ReplaceAll(fileText, *oldTextPtr, *newTextPtr)

	err = os.WriteFile(*pathPtr, []byte(newContent), 0644)
	if err != nil {
		printResult(false, "", fmt.Sprintf("Failed to write file: %v", err))
		return
	}

	printResult(true, "File modified successfully", "")
}

func printResult(success bool, msg string, errStr string) {
	res := ModifyResult{
		Success: success,
		Message: msg,
		Error:   errStr,
	}
	data, _ := json.Marshal(res)
	fmt.Println(string(data))
}
