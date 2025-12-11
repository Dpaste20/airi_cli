package main

import (
	"encoding/json"
	"fmt"
	"math"
	"syscall"
)

type DiskSpaceInfo struct {
	TotalGB      float64 `json:"total_GB"`
	UsedGB       float64 `json:"used_GB"`
	FreeGB       float64 `json:"free_GB"`
	UsagePercent float64 `json:"usage_percent"`
	Error        string  `json:"error,omitempty"`
}

func main() {
	info := getDiskSpace("/")
	jsonData, err := json.Marshal(info)
	if err != nil {
		fmt.Println(`{"error": "Failed to encode JSON response"}`)
		return
	}
	fmt.Println(string(jsonData))
}

func getDiskSpace(path string) DiskSpaceInfo {
	var stat syscall.Statfs_t
	err := syscall.Statfs(path, &stat)
	if err != nil {
		return DiskSpaceInfo{Error: fmt.Sprintf("Failed to get disk usage: %v", err)}
	}

	totalBytes := float64(stat.Blocks) * float64(stat.Bsize)
	freeBytes := float64(stat.Bfree) * float64(stat.Bsize)
	usedBytes := totalBytes - freeBytes

	gb := math.Pow(1024, 3)

	totalGB := round(totalBytes/gb, 2)
	usedGB := round(usedBytes/gb, 2)
	freeGB := round(freeBytes/gb, 2)

	var usagePercent float64
	if totalBytes > 0 {
		usagePercent = round((usedBytes/totalBytes)*100, 2)
	}

	return DiskSpaceInfo{
		TotalGB:      totalGB,
		UsedGB:       usedGB,
		FreeGB:       freeGB,
		UsagePercent: usagePercent,
	}
}

func round(val float64, precision int) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}
