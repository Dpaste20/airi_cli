package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"
)

func main() {

	braveCmd := exec.Command("brave-browser", "--remote-debugging-port=9222", "--user-data-dir=airi_browse_dir")

	fmt.Println("1. Launching Brave Browser...")

	err := braveCmd.Start()
	if err != nil {
		log.Fatalf("Failed to launch Brave: %v", err)
	}
	fmt.Printf("   Brave started (PID: %d). Remote debugging on port 9222.\n", braveCmd.Process.Pid)

	fmt.Println("2. Waiting 3 seconds for debugging port to initialize...")
	time.Sleep(3 * time.Second)

	fmt.Println("3. Connecting agent-browser...")

	agentCmd := exec.Command("agent-browser", "connect", "9222")

	agentCmd.Stdout = os.Stdout
	agentCmd.Stderr = os.Stderr
	agentCmd.Stdin = os.Stdin

	if err := agentCmd.Run(); err != nil {
		log.Fatalf("agent-browser failed: %v", err)
	}

	fmt.Println("Agent process finished.")
}
