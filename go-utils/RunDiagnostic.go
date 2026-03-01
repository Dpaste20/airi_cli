package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
)

type DiagnosticReport struct {
	Status      string `json:"status"`
	CPUUsage    string `json:"cpu_usage"`
	RAMUsage    string `json:"ram_usage"`
	DiskSpace   string `json:"disk_space"`
	Thermals    string `json:"thermals"`
	NetworkPing string `json:"network_ping"`
}

func main() {
	report := DiagnosticReport{Status: "Healthy"}
	var wg sync.WaitGroup
	var mu sync.Mutex

	hasWarning := false

	wg.Add(5)

	go func() {
		defer wg.Done()
		var info syscall.Sysinfo_t
		if err := syscall.Sysinfo(&info); err == nil {
			unit := uint64(info.Unit)
			total := uint64(info.Totalram) * unit
			free := uint64(info.Freeram) * unit
			buffers := uint64(info.Bufferram) * unit
			available := free + buffers
			used := total - available

			usedPercent := (float64(used) / float64(total)) * 100
			totalGB := float64(total) / math.Pow(1024, 3)
			availableGB := float64(available) / math.Pow(1024, 3)

			mu.Lock()
			report.RAMUsage = fmt.Sprintf("%.2f GB / %.2f GB (%.1f%%)", (totalGB - availableGB), totalGB, usedPercent)
			if usedPercent > 90 {
				hasWarning = true
				report.RAMUsage += " - WARNING: High RAM usage"
			}
			mu.Unlock()
		} else {
			report.RAMUsage = "Error reading RAM"
		}
	}()

	go func() {
		defer wg.Done()
		content, err := os.ReadFile("/proc/loadavg")
		if err == nil {
			parts := strings.Fields(string(content))
			if len(parts) >= 1 {
				load1, _ := strconv.ParseFloat(parts[0], 64)
				cores := float64(runtime.NumCPU())

				mu.Lock()
				report.CPUUsage = fmt.Sprintf("Load Avg (1m): %.2f | Cores: %d", load1, int(cores))
				if load1 > cores {
					hasWarning = true
					report.CPUUsage += " - WARNING: High CPU Pressure"
				}
				mu.Unlock()
			}
		} else {
			report.CPUUsage = "Error reading CPU load"
		}
	}()

	go func() {
		defer wg.Done()
		var stat syscall.Statfs_t
		if err := syscall.Statfs("/", &stat); err == nil {
			total := float64(stat.Blocks) * float64(stat.Bsize)
			free := float64(stat.Bavail) * float64(stat.Bsize)
			used := total - free
			usedPercent := (used / total) * 100

			totalGB := total / math.Pow(1024, 3)
			freeGB := free / math.Pow(1024, 3)

			mu.Lock()
			report.DiskSpace = fmt.Sprintf("%.2f GB Free of %.2f GB (%.1f%% Used)", freeGB, totalGB, usedPercent)
			if usedPercent > 90 {
				hasWarning = true
				report.DiskSpace += " - WARNING: Low Disk Space"
			}
			mu.Unlock()
		} else {
			report.DiskSpace = "Error reading Disk"
		}
	}()

	go func() {
		defer wg.Done()
		content, err := os.ReadFile("/sys/class/thermal/thermal_zone0/temp")
		if err == nil {
			tempStr := strings.TrimSpace(string(content))
			tempMilliC, err := strconv.Atoi(tempStr)
			if err == nil {
				tempC := float64(tempMilliC) / 1000.0
				mu.Lock()
				report.Thermals = fmt.Sprintf("%.1f°C", tempC)
				if tempC > 80.0 {
					hasWarning = true
					report.Thermals += " - WARNING: High Temperature"
				}
				mu.Unlock()
			}
		} else {
			report.Thermals = "Sensor not available"
		}
	}()

	go func() {
		defer wg.Done()
		out, err := exec.Command("ping", "-c", "1", "-W", "1", "8.8.8.8").Output()
		mu.Lock()
		defer mu.Unlock()

		if err != nil {
			report.NetworkPing = "Disconnected (Ping Failed)"
			hasWarning = true
		} else {
			output := string(out)
			if strings.Contains(output, "time=") {
				parts := strings.Split(output, "time=")
				if len(parts) > 1 {
					timePart := strings.Split(parts[1], " ")[0]
					report.NetworkPing = fmt.Sprintf("Connected (%s ms)", timePart)
				} else {
					report.NetworkPing = "Connected"
				}
			} else {
				report.NetworkPing = "Connected"
			}
		}
	}()

	wg.Wait()

	if hasWarning {
		report.Status = "Warning - Check Metrics"
	}

	jsonData, _ := json.MarshalIndent(report, "", "  ")
	fmt.Println(string(jsonData))
}
