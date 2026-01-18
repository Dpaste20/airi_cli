package main

import (
	"fmt"
	"os/exec"
	"time"
)

func main() {

	time.Sleep(3 * time.Second)

	cmd := exec.Command("shutdown", "-h", "now")

	output, err := cmd.CombinedOutput()
	if err != nil {
		printError(fmt.Sprintf("Failed to shutdown: %v. Output: %s", err, string(output)))
		return
	}

	fmt.Println("[{\"status\": \"success\", \"message\": \"Airi and system shutdown initiated after 5 seconds.\"}]")
}

func printError(msg string) {
	fmt.Printf("[{\"error\": \"%s\"}]\n", msg)
}
