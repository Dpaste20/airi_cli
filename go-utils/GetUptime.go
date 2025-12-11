package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type UptimeInfo struct {
	UptimeSeconds  int    `json:"uptime_seconds"`
	UptimeReadable string `json:"uptime_readable"`
	Error          string `json:"error,omitempty"`
}

func main() {
	info := getUptime()

	jsonData, err := json.Marshal(info)
	if err != nil {
		fmt.Println(`{"error": "Failed to encode JSON response"}`)
		return
	}
	fmt.Println(string(jsonData))
}

func getUptime() UptimeInfo {
	uptimeFile := "/proc/uptime"

	if _, err := os.Stat(uptimeFile); os.IsNotExist(err) {
		return UptimeInfo{Error: "Uptime information not available"}
	}

	content, err := os.ReadFile(uptimeFile)
	if err != nil {
		return UptimeInfo{Error: fmt.Sprintf("Failed to read uptime file: %v", err)}
	}

	parts := strings.Fields(string(content))
	if len(parts) == 0 {
		return UptimeInfo{Error: "Invalid uptime file format"}
	}

	uptimeSecondsFloat, err := strconv.ParseFloat(parts[0], 64)
	if err != nil {
		return UptimeInfo{Error: fmt.Sprintf("Failed to parse uptime: %v", err)}
	}

	uptimeSeconds := int(uptimeSecondsFloat)

	days := uptimeSeconds / 86400
	hours := (uptimeSeconds % 86400) / 3600
	minutes := (uptimeSeconds % 3600) / 60
	seconds := uptimeSeconds % 60

	readable := fmt.Sprintf("%dd %dh %dm %ds", days, hours, minutes, seconds)

	return UptimeInfo{
		UptimeSeconds:  uptimeSeconds,
		UptimeReadable: readable,
	}
}
