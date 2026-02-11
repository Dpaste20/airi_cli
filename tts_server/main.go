package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"syscall"

	"gopkg.in/yaml.v3"
)

// Config struct maps to the keys in config.yaml
type Config struct {
	Host  string `yaml:"host"`
	Port  string `yaml:"port"`
	Voice string `yaml:"voice"`
}

func main() {
	// 1. Read the config file
	configFile, err := os.ReadFile("config.yaml")
	if err != nil {
		log.Fatalf("Error reading config.yaml: %v", err)
	}

	// 2. Unmarshal (parse) the YAML data into the struct
	var cfg Config
	err = yaml.Unmarshal(configFile, &cfg)
	if err != nil {
		log.Fatalf("Error parsing config.yaml: %v", err)
	}

	// Set defaults if missing
	if cfg.Host == "" {
		cfg.Host = "localhost"
	}
	if cfg.Port == "" {
		cfg.Port = "8000"
	}
	if cfg.Voice == "" {
		cfg.Voice = "cosette"
	}

	// 3. Create a context that listens for interrupt signals (Ctrl+C or SIGTERM)
	// This ensures we can clean up the sub-process when the main app is stopped.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// 4. Construct the command using CommandContext
	// When 'ctx' is canceled (by Ctrl+C), this command will be sent a kill signal automatically.
	cmd := exec.CommandContext(ctx, "uvx", "pocket-tts", "serve",
		"--host", cfg.Host,
		"--port", cfg.Port,
		"--voice", cfg.Voice,
	)

	// Connect stdout/stderr to see logs
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	fmt.Printf("Starting Pocket-TTS on %s:%s with voice '%s'...\n", cfg.Host, cfg.Port, cfg.Voice)

	// 5. Start the command (non-blocking)
	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start command: %v", err)
	}

	// 6. Wait for the command to finish or the context to be canceled
	err = cmd.Wait()

	// Check if the exit was caused by our signal interrupt (graceful shutdown)
	if ctx.Err() == context.Canceled {
		fmt.Println("\nReceived interrupt signal. Gracefully shutting down Pocket-TTS...")
	} else if err != nil {
		// If it crashed for another reason
		fmt.Printf("Process finished with error: %v\n", err)
		os.Exit(1)
	}
}
