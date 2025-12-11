package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type BatteryInfo struct {
	BatteryID  string `json:"battery_id"`
	Percentage int    `json:"percentage"`
	Status     string `json:"status"`
	Error      string `json:"error,omitempty"`
}

func main() {
	info := getBatteryStatus()
	jsonData, err := json.Marshal(info)
	if err != nil {
		fmt.Println(`{"error": "Failed to encode JSON response"}`)
		return
	}
	fmt.Println(string(jsonData))
}

func getBatteryStatus() BatteryInfo {
	powerSupplyPath := "/sys/class/power_supply"

	if _, err := os.Stat(powerSupplyPath); os.IsNotExist(err) {
		return BatteryInfo{Error: "Battery information not available"}
	}

	entries, err := os.ReadDir(powerSupplyPath)
	if err != nil {
		return BatteryInfo{Error: fmt.Sprintf("Failed to read directory: %v", err)}
	}

	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "BAT") {
			batteryPath := filepath.Join(powerSupplyPath, entry.Name())

			capacityBytes, err := os.ReadFile(filepath.Join(batteryPath, "capacity"))
			if err != nil {
				continue
			}
			capacityStr := strings.TrimSpace(string(capacityBytes))
			capacity, err := strconv.Atoi(capacityStr)
			if err != nil {
				continue
			}

			statusBytes, err := os.ReadFile(filepath.Join(batteryPath, "status"))
			if err != nil {
				continue
			}
			status := strings.TrimSpace(string(statusBytes))

			return BatteryInfo{
				BatteryID:  entry.Name(),
				Percentage: capacity,
				Status:     status,
			}
		}
	}

	return BatteryInfo{Error: "No battery found"}
}
