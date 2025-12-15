package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
)

type AppResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
	Pid     int    `json:"pid,omitempty"`
	Error   string `json:"error,omitempty"`
}

func main() {
	if len(os.Args) < 2 {
		sendResponse(AppResponse{Status: "error", Error: "No target provided"})
		return
	}

	target := os.Args[1]
	var cmd *exec.Cmd
	var method string

	if isURL(target) || fileExists(target) {
		method = "xdg-open"
		cmd = exec.Command("xdg-open", target)
	} else {
		resolvedCmd, matchName, found := resolveAppCommand(target)

		if !found {
			path, err := exec.LookPath(target)
			if err != nil {
				sendResponse(AppResponse{
					Status: "error",
					Error:  fmt.Sprintf("Could not find application or command matching '%s'.", target),
				})
				return
			}
			resolvedCmd = path
			method = "direct-command"
		} else {
			method = fmt.Sprintf("smart-match (%s)", matchName)
		}

		parts := strings.Fields(resolvedCmd)
		head := parts[0]
		args := parts[1:]
		cmd = exec.Command(head, args...)
	}

	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	cmd.Stdout = nil
	cmd.Stderr = nil

	if err := cmd.Start(); err != nil {
		sendResponse(AppResponse{
			Status: "error",
			Error:  fmt.Sprintf("Failed to launch using %s: %v", method, err),
		})
		return
	}

	cmd.Process.Release()

	sendResponse(AppResponse{
		Status:  "success",
		Message: fmt.Sprintf("Opened '%s' using %s", target, method),
		Pid:     cmd.Process.Pid,
	})
}

func resolveAppCommand(query string) (string, string, bool) {
	query = strings.ToLower(query)

	dirs := []string{
		"/usr/share/applications",
		os.ExpandEnv("$HOME/.local/share/applications"),
		"/var/lib/snapd/desktop/applications",
		"/usr/local/share/applications",
	}

	var bestMatchCmd string
	var bestMatchName string
	bestScore := -1

	for _, dir := range dirs {
		files, err := os.ReadDir(dir)
		if err != nil {
			continue
		}

		for _, file := range files {
			if strings.HasSuffix(file.Name(), ".desktop") {
				path := filepath.Join(dir, file.Name())
				name, execCmd := parseDesktopFile(path)

				if name == "" || execCmd == "" {
					continue
				}

				lowerName := strings.ToLower(name)
				lowerExec := strings.ToLower(execCmd)

				currentScore := -1

				if lowerName == query || lowerExec == query {
					currentScore = 2
				} else if strings.HasPrefix(lowerName, query) {
					currentScore = 1
				} else if strings.Contains(lowerName, query) || strings.Contains(lowerExec, query) {
					currentScore = 0
				}

				if currentScore > bestScore {
					bestScore = currentScore
					bestMatchCmd = execCmd
					bestMatchName = name
					if bestScore == 2 {
						return bestMatchCmd, bestMatchName, true
					}
				}
			}
		}
	}

	if bestScore > -1 {
		return bestMatchCmd, bestMatchName, true
	}

	return "", "", false
}

func parseDesktopFile(path string) (string, string) {
	file, err := os.Open(path)
	if err != nil {
		return "", ""
	}
	defer file.Close()

	var name, execCmd string
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if strings.HasPrefix(line, "Name=") && name == "" {
			name = line[5:]
		}
		if strings.HasPrefix(line, "Exec=") && execCmd == "" {
			execCmd = line[5:]
		}
		if name != "" && execCmd != "" {
			break
		}
	}

	if execCmd != "" {
		parts := strings.Split(execCmd, " ")
		cleanCmd := []string{}
		for _, part := range parts {
			if !strings.HasPrefix(part, "%") {
				cleanCmd = append(cleanCmd, part)
			}
		}
		execCmd = strings.Join(cleanCmd, " ")
	}

	return name, execCmd
}

func isURL(str string) bool {
	u, err := url.Parse(str)
	return err == nil && (u.Scheme == "http" || u.Scheme == "https")
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func sendResponse(resp AppResponse) {
	jsonData, err := json.Marshal(resp)
	if err != nil {
		fmt.Println(`{"status": "error", "error": "JSON encoding failed"}`)
		return
	}
	fmt.Println(string(jsonData))
}
