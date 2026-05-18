package main

import (
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"
)

func main() {
	macrosFile := "macros.go"
	yamlFile := "../knownCommands.yaml"

	coreCommands := []string{
		"/help",
		"/attach",
		"/detach",
		"/save-session",
		"/resume-session",
		"/list-sessions",
		"/tools-list",
		"/web-ui",
		"/lore",
	}

	data, err := os.ReadFile(macrosFile)
	if err != nil {
		log.Fatalf("❌ Failed to read %s: %v", macrosFile, err)
	}

	re := regexp.MustCompile(`"(/\w[\w-]*)"\s*:`)
	matches := re.FindAllStringSubmatch(string(data), -1)

	macroSet := make(map[string]bool)
	for _, match := range matches {
		if len(match) > 1 {
			cmd := match[1]
			isCore := false
			for _, core := range coreCommands {
				if core == cmd {
					isCore = true
					break
				}
			}
			if !isCore {
				macroSet[cmd] = true
			}
		}
	}

	var sortedMacros []string
	for cmd := range macroSet {
		sortedMacros = append(sortedMacros, cmd)
	}
	sort.Strings(sortedMacros)

	var sb strings.Builder
	sb.WriteString("commands:\n")

	for _, cmd := range coreCommands {
		sb.WriteString(fmt.Sprintf("  - %s\n", cmd))
	}

	for _, cmd := range sortedMacros {
		sb.WriteString(fmt.Sprintf("  - %s\n", cmd))
	}

	err = os.WriteFile(yamlFile, []byte(sb.String()), 0644)
	if err != nil {
		log.Fatalf("❌ Failed to write %s: %v", yamlFile, err)
	}

	totalCommands := len(coreCommands) + len(sortedMacros)
	fmt.Printf("Successfully synced %d commands to %s\n", totalCommands, yamlFile)
}
