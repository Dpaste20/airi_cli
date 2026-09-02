package commands

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

func LaunchWebUI() (Result, bool) {

	cwd, err := os.Getwd()
	if err != nil {
		cwd = "."
	}

	targetPath := filepath.Join(cwd, "commands", "LauchWebUI")

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
