package commands

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

func findLauncher() (string, error) {
	launcherName := "LauchWebUI"

	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		for {
			candidate := filepath.Join(dir, "commands", launcherName)
			if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
				return candidate, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("could not locate %s: %v", launcherName, err)
	}

	dir := cwd
	for {
		candidate := filepath.Join(dir, "commands", launcherName)
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return "", fmt.Errorf("could not locate %s in any parent directory", launcherName)
}

func LaunchWebUI() (Result, bool) {

	targetPath, err := findLauncher()
	if err != nil {
		return Result{
			IsError:         true,
			ViewportMessage: fmt.Sprintf("❌ **Failed to launch Web UI:**\n`%v`", err),
			Notification:    "Launch failed",
		}, true
	}

	cmd := exec.Command(targetPath)

	if err := cmd.Start(); err != nil {
		return Result{
			IsError:         true,
			ViewportMessage: fmt.Sprintf("❌ **Failed to launch Web UI:**\n`%v`\n\n*Attempted to run:* `%s`", err, targetPath),
			Notification:    "Launch failed",
		}, true
	}

	go func() {
		_ = cmd.Wait()
	}()

	return Result{
		ViewportMessage: fmt.Sprintf("Web UI lauched , see at http://localhost:5173/ "),
		Notification:    "Web UI Started",
	}, true
}
