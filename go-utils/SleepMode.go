package main

import (
	"fmt"
	"os/exec"
	"time"
)

func main() {

	time.Sleep(5 * time.Second)

	cmd := exec.Command("systemctl", "suspend")

	output, err := cmd.CombinedOutput()
	if err != nil {
		printError(fmt.Sprintf("Failed to enter sleep mode: %v. Output: %s", err, string(output)))
		return
	}

	fmt.Println("[{\"status\": \"success\", \"message\": \"System sleep mode initiated.\"}]")
}

func printError(msg string) {
	fmt.Printf("[{\"error\": \"%s\"}]\n", msg)
}
