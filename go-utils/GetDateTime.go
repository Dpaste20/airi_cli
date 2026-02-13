package main

import (
	"encoding/json"
	"fmt"
	"time"
)

type DateTimeInfo struct {
	CurrentTime string `json:"current_time"`
	CurrentDate string `json:"current_date"`
	DayOfWeek   string `json:"day_of_week"`
	Timezone    string `json:"timezone"`
	IsoDateTime string `json:"iso_datetime"`
}

func main() {
	now := time.Now()

	info := DateTimeInfo{
		CurrentTime: now.Format("15:04:05"),

		CurrentDate: now.Format("2006-01-02"),
		DayOfWeek:   now.Weekday().String(),
		Timezone:    now.Location().String(),
		// Formats as 2026-02-13T15:04:05+05:30 (Includes timezone offset!)
		IsoDateTime: now.Format(time.RFC3339),
	}

	jsonData, err := json.Marshal(info)
	if err != nil {
		fmt.Println(`{"error": "Failed to encode time data"}`)
		return
	}
	fmt.Println(string(jsonData))
}
