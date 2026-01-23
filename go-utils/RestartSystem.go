package main

import (
	"fmt"
	"os/exec"
	"time"
)

func main() {

	time.Sleep(5 * time.Second)

	cmd := exec.Command("shutdown", "-r", "now")

	output, err := cmd.CombinedOutput()
	if err != nil {
		printError(fmt.Sprintf("Failed to restart: %v. Output: %s", err, string(output)))
		return
	}

	fmt.Println("[{\"status\": \"success\", \"message\": \"Airi and system restart initiated after 5 seconds.\"}]")
}

func printError(msg string) {
	fmt.Printf("[{\"error\": \"%s\"}]\n", msg)
}
