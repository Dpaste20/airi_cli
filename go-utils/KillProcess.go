package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"syscall"
)

type Response struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

func main() {
	pidPtr := flag.Int("pid", 0, "Process ID to terminate")
	forcePtr := flag.Bool("force", false, "Force kill (SIGKILL/-9)")
	flag.Parse()

	if *pidPtr <= 0 {
		printOutput("error", "Invalid or missing PID")
		return
	}

	pid := *pidPtr

	proc, err := os.FindProcess(pid)
	if err != nil {
		printOutput("error", fmt.Sprintf("Failed to find process: %v", err))
		return
	}

	if err := proc.Signal(syscall.Signal(0)); err != nil {
		if err == os.ErrProcessDone {
			printOutput("error", fmt.Sprintf("Process %d has already finished", pid))
		} else {
			printOutput("error", fmt.Sprintf("Process %d not found or access denied", pid))
		}
		return
	}

	var sig syscall.Signal
	signalName := "terminated"
	if *forcePtr {
		sig = syscall.SIGKILL
		signalName = "forcefully terminated"
	} else {
		sig = syscall.SIGTERM
	}

	err = proc.Signal(sig)
	if err != nil {
		printOutput("error", fmt.Sprintf("Failed to kill process %d: %v", pid, err))
		return
	}

	printOutput("success", fmt.Sprintf("Process %d %s.", pid, signalName))
}

func printOutput(status, message string) {
	resp := Response{
		Status:  status,
		Message: message,
	}
	jsonData, _ := json.Marshal(resp)
	fmt.Println(string(jsonData))
}
