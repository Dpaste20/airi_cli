package main

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

type CronJob struct {
	ID       string `json:"id"`
	Schedule string `json:"schedule"`
	Command  string `json:"command"`
	Raw      string `json:"raw"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println(`{"error": "Usage: CronManager <list|add|remove> [args]"}`)
		os.Exit(1)
	}

	action := os.Args[1]

	switch action {
	case "list":
		listCronJobs()
	case "add":
		if len(os.Args) < 4 {
			fmt.Println(`{"error": "Usage: CronManager add <schedule> <command>"}`)
			os.Exit(1)
		}
		addCronJob(os.Args[2], os.Args[3])
	case "remove":
		if len(os.Args) < 3 {
			fmt.Println(`{"error": "Usage: CronManager remove <id>"}`)
			os.Exit(1)
		}
		removeCronJob(os.Args[2])
	default:
		fmt.Printf(`{"error": "Unknown action: %s"}\n`, action)
		os.Exit(1)
	}
}

func getCrontabContent() ([]string, error) {
	cmd := exec.Command("crontab", "-l")
	output, err := cmd.Output()

	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			if exitError.ExitCode() == 1 {
				return []string{}, nil
			}
		}
		return nil, err
	}

	lines := strings.Split(string(output), "\n")
	var cleaned []string
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			cleaned = append(cleaned, line)
		}
	}
	return cleaned, nil
}

func setCrontabContent(lines []string) error {
	content := strings.Join(lines, "\n") + "\n"
	cmd := exec.Command("crontab", "-")
	cmd.Stdin = strings.NewReader(content)
	return cmd.Run()
}

func generateID(line string) string {
	hash := md5.Sum([]byte(line))
	return hex.EncodeToString(hash[:])[:8]
}

func listCronJobs() {
	lines, err := getCrontabContent()
	if err != nil {
		fmt.Printf(`{"error": "Failed to read crontab: %v"}\n`, err)
		os.Exit(1)
	}

	var jobs []CronJob
	for _, line := range lines {

		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) >= 6 {

			schedule := strings.Join(parts[:5], " ")
			command := strings.Join(parts[5:], " ")

			jobs = append(jobs, CronJob{
				ID:       generateID(line),
				Schedule: schedule,
				Command:  command,
				Raw:      line,
			})
		}
	}

	jsonData, _ := json.Marshal(jobs)
	fmt.Println(string(jsonData))
}

func addCronJob(schedule, command string) {
	lines, err := getCrontabContent()
	if err != nil {
		fmt.Printf(`{"error": "Failed to read crontab: %v"}\n`, err)
		os.Exit(1)
	}

	if strings.Contains(command, "\n") {
		fmt.Println(`{"error": "Command cannot contain newlines"}`)
		os.Exit(1)
	}

	newLine := fmt.Sprintf("%s %s", schedule, command)

	for _, line := range lines {
		if line == newLine {
			fmt.Println(`{"status": "skipped", "message": "Job already exists"}`)
			return
		}
	}

	lines = append(lines, newLine)

	if err := setCrontabContent(lines); err != nil {
		fmt.Printf(`{"error": "Failed to write crontab: %v"}\n`, err)
		os.Exit(1)
	}

	fmt.Printf(`{"status": "success", "id": "%s", "message": "Job added"}\n`, generateID(newLine))
}

func removeCronJob(targetID string) {
	lines, err := getCrontabContent()
	if err != nil {
		fmt.Printf(`{"error": "Failed to read crontab: %v"}\n`, err)
		os.Exit(1)
	}

	var newLines []string
	found := false

	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			newLines = append(newLines, line)
			continue
		}

		currentID := generateID(line)
		if currentID == targetID {
			found = true
			continue
		}
		newLines = append(newLines, line)
	}

	if !found {
		fmt.Printf(`{"error": "Job with ID %s not found"}\n`, targetID)
		os.Exit(1)
	}

	if err := setCrontabContent(newLines); err != nil {
		fmt.Printf(`{"error": "Failed to update crontab: %v"}\n`, err)
		os.Exit(1)
	}

	fmt.Println(`{"status": "success", "message": "Job removed"}`)
}
