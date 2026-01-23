package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

type UrlOpenResponse struct {
	URL     string `json:"url,omitempty"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
}

func openUrl(targetUrl string) error {
	url := strings.TrimSpace(targetUrl)

	// Add https:// if no protocol is specified
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		url = "https://" + url
	}

	var cmd *exec.Cmd

	// Determine the command based on the operating system
	switch runtime.GOOS {
	case "linux":
		cmd = exec.Command("xdg-open", url)
	case "darwin": // macOS
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		return fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}

	// Execute the command
	err := cmd.Start()
	if err != nil {
		return fmt.Errorf("failed to open URL: %v", err)
	}

	return nil
}

func main() {
	url := flag.String("url", "", "URL to open in default browser")
	flag.Parse()

	var response UrlOpenResponse

	if *url == "" {
		response.Error = "No URL provided"
		jsonOutput, _ := json.Marshal(response)
		fmt.Println(string(jsonOutput))
		return
	}

	err := openUrl(*url)
	if err != nil {
		response.Error = fmt.Sprintf("OpenUrl failed: %v", err)
	} else {
		response.URL = *url
		if !strings.HasPrefix(*url, "http://") && !strings.HasPrefix(*url, "https://") {
			response.URL = "https://" + *url
		}
		response.Message = "URL opened successfully in default browser"
	}

	jsonOutput, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		fmt.Printf("{\"error\": \"JSON marshal error: %v\"}\n", err)
		return
	}

	fmt.Println(string(jsonOutput))
}
