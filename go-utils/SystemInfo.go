package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

type SystemInfo struct {
	Host          HostInfo   `json:"host"`
	Memory        MemoryInfo `json:"memory"`
	CPU           CPUInfo    `json:"cpu"`
	UptimeSeconds uint64     `json:"uptime_seconds"`
	UptimeHuman   string     `json:"uptime_human"`
	Error         string     `json:"error,omitempty"`
}

type HostInfo struct {
	Hostname string `json:"hostname"`
	OS       string `json:"os"`
	Arch     string `json:"arch"`
	Kernel   string `json:"kernel_version"`
}

type MemoryInfo struct {
	TotalGB     float64 `json:"total_GB"`
	FreeGB      float64 `json:"free_GB"`
	AvailableGB float64 `json:"available_GB"`
	UsedPercent float64 `json:"used_percent"`
}

type CPUInfo struct {
	Model       string    `json:"model"`
	Cores       int       `json:"cores"`
	LoadAverage []float64 `json:"load_average"`
}

func main() {
	info := getSystemInfo()
	jsonData, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		fmt.Println(`{"error": "Failed to encode JSON response"}`)
		return
	}
	fmt.Println(string(jsonData))
}

func getSystemInfo() SystemInfo {
	var sysInfo SystemInfo
	var err error

	sysInfo.Host.Hostname, _ = os.Hostname()
	sysInfo.Host.OS = runtime.GOOS
	sysInfo.Host.Arch = runtime.GOARCH
	sysInfo.Host.Kernel = getKernelVersion()

	var info syscall.Sysinfo_t
	err = syscall.Sysinfo(&info)
	if err == nil {
		sysInfo.UptimeSeconds = uint64(info.Uptime)
		sysInfo.UptimeHuman = formatUptime(info.Uptime)

		unit := uint64(info.Unit)
		totalBytes := uint64(info.Totalram) * unit
		freeBytes := uint64(info.Freeram) * unit
		bufferBytes := uint64(info.Bufferram) * unit

		availableBytes := freeBytes + bufferBytes
		usedBytes := totalBytes - availableBytes

		gb := math.Pow(1024, 3)
		sysInfo.Memory.TotalGB = round(float64(totalBytes)/gb, 2)
		sysInfo.Memory.FreeGB = round(float64(freeBytes)/gb, 2)
		sysInfo.Memory.AvailableGB = round(float64(availableBytes)/gb, 2)

		if totalBytes > 0 {
			sysInfo.Memory.UsedPercent = round((float64(usedBytes)/float64(totalBytes))*100, 2)
		}
	} else {
		sysInfo.Error = fmt.Sprintf("Syscall failed: %v", err)
	}

	sysInfo.CPU.Cores = runtime.NumCPU()
	sysInfo.CPU.Model = getCPUModel()
	sysInfo.CPU.LoadAverage = getLoadAvg()

	return sysInfo
}

func getKernelVersion() string {
	content, err := os.ReadFile("/proc/sys/kernel/osrelease")
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(content))
}

func getCPUModel() string {
	file, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return "unknown"
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "model name") {
			parts := strings.Split(line, ":")
			if len(parts) > 1 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "unknown"
}

func getLoadAvg() []float64 {
	content, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return []float64{0, 0, 0}
	}
	parts := strings.Fields(string(content))
	if len(parts) < 3 {
		return []float64{0, 0, 0}
	}

	loads := make([]float64, 3)
	for i := 0; i < 3; i++ {
		val, _ := strconv.ParseFloat(parts[i], 64)
		loads[i] = val
	}
	return loads
}

func formatUptime(uptime int64) string {
	days := uptime / (60 * 60 * 24)
	hours := (uptime % (60 * 60 * 24)) / (60 * 60)
	minutes := (uptime % (60 * 60)) / 60
	return fmt.Sprintf("%dd %dh %dm", days, hours, minutes)
}

func round(val float64, precision int) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}
