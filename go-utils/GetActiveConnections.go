package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"strings"

	"github.com/shirou/gopsutil/v3/net"
	"github.com/shirou/gopsutil/v3/process"
)

type ConnectionInfo struct {
	Type    string `json:"type"`
	Local   string `json:"local"`
	Remote  string `json:"remote"`
	Status  string `json:"status"`
	PID     int32  `json:"pid"`
	Program string `json:"program"`
}

func main() {
	limitPtr := flag.Int("limit", 20, "Max number of connections to return")
	statePtr := flag.String("state", "", "Filter by state (e.g., ESTABLISHED, LISTEN)")
	flag.Parse()

	connections, err := net.Connections("inet")
	if err != nil {
		printError(fmt.Sprintf("Failed to get connections: %v", err))
		return
	}

	var results []ConnectionInfo
	procNames := make(map[int32]string)

	count := 0
	for _, conn := range connections {
		if count >= *limitPtr {
			break
		}

		if *statePtr != "" && !strings.EqualFold(conn.Status, *statePtr) {
			continue
		}

		cType := "TCP"
		if conn.Type == 2 {
			cType = "UDP"
		}

		localAddr := fmt.Sprintf("%s:%d", conn.Laddr.IP, conn.Laddr.Port)
		remoteAddr := fmt.Sprintf("%s:%d", conn.Raddr.IP, conn.Raddr.Port)

		procName := ""
		if conn.Pid > 0 {
			if name, ok := procNames[conn.Pid]; ok {
				procName = name
			} else {
				p, err := process.NewProcess(conn.Pid)
				if err == nil {
					n, _ := p.Name()
					procName = n
					procNames[conn.Pid] = n
				}
			}
		}

		results = append(results, ConnectionInfo{
			Type:    cType,
			Local:   localAddr,
			Remote:  remoteAddr,
			Status:  conn.Status,
			PID:     conn.Pid,
			Program: procName,
		})

		count++
	}

	if len(results) == 0 {
		fmt.Println("[]")
	} else {
		jsonData, err := json.MarshalIndent(results, "", "  ")
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
