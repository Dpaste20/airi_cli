package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
)

type CloseResponse struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
}

func main() {
	if len(os.Args) < 2 {
		respond(CloseResponse{
			Status: "error",
			Error:  "No application name provided",
		})
		return
	}

	appName := os.Args[1]

	err := exec.Command("pkill", "-i", appName).Run()
	if err != nil {
		respond(CloseResponse{
			Status: "error",
			Error:  fmt.Sprintf("No running application named '%s'", appName),
		})
		return
	}

	respond(CloseResponse{
		Status:  "success",
		Message: fmt.Sprintf("Closed application '%s'", appName),
	})
}

func respond(r CloseResponse) {
	out, _ := json.Marshal(r)
	fmt.Println(string(out))
}
