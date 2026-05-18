package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

func main() {

	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		log.Fatal("Failed to get current script path")
	}

	scriptDir := filepath.Dir(currentFile)
	targetDir := filepath.Clean(filepath.Join(scriptDir, "..", "web-ui", "web-ui-site"))

	targetURL := "http://localhost:5173"

	cmd := exec.Command("npm", "run", "dev")
	cmd.Dir = targetDir

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	fmt.Printf("🚀 Launching 'npm run dev' in %s...\n", targetDir)
	fmt.Println(strings.Repeat("-", 70))

	go func() {

		time.Sleep(3 * time.Second)
		fmt.Printf("\nOpening browser to %s\n", targetURL)

		err := exec.Command("xdg-open", targetURL).Start()
		if err != nil {
			fmt.Printf("Could not open browser automatically: %v\n", err)
		}
	}()

	err := cmd.Run()
	if err != nil {
		log.Fatalf("\n❌ Command finished with error: %v", err)
	}
}
