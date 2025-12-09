package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"sort"

	"github.com/shirou/gopsutil/v3/process"
)

type ProcessInfo struct {
	PID     int32   `json:"pid"`
	User    string  `json:"user"`
	CPU     float64 `json:"cpu"`
	Memory  float32 `json:"memory"`
	Command string  `json:"command"`
}

func main() {

	limitPtr := flag.Int("limit", 10, "The number of processes to return")
	flag.Parse()

	procs, err := process.Processes()
	if err != nil {
		printError(fmt.Sprintf("Failed to list processes: %v", err))
		return
	}

	var procList []ProcessInfo

	for _, p := range procs {
		cpu, err := p.CPUPercent()
		if err != nil {
			continue
		}

		if cpu == 0.0 {
			continue
		}

		mem, _ := p.MemoryPercent()
		username, _ := p.Username()

		cmd, err := p.Cmdline()
		if err != nil || cmd == "" {
			cmd, _ = p.Name()
		}

		procList = append(procList, ProcessInfo{
			PID:     p.Pid,
			User:    username,
			CPU:     cpu,
			Memory:  mem,
			Command: cmd,
		})
	}

	sort.Slice(procList, func(i, j int) bool {
		return procList[i].CPU > procList[j].CPU
	})

	if len(procList) > *limitPtr {
		procList = procList[:*limitPtr]
	}

	if len(procList) == 0 {
		fmt.Println("[]")
	} else {
		jsonData, err := json.Marshal(procList)
		if err != nil {
			printError(fmt.Sprintf("JSON marshal error: %v", err))
			return
		}
		fmt.Println(string(jsonData))
	}
}

func printError(msg string) {
	fmt.Printf("[{\"error\": \"%s\"}]\n", msg)
}
