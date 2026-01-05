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
}

func main() {
	now := time.Now()
	info := DateTimeInfo{
		CurrentTime: now.Format(time.RFC3339),
		CurrentDate: now.Format(time.DateOnly), // YYYY-MM-DD
		DayOfWeek:   now.Weekday().String(),
		Timezone:    now.Format("MST"),
	}

	jsonData, err := json.Marshal(info)
	if err != nil {
		fmt.Println(`{"error": "Failed to encode time data"}`)
		return
	}
	fmt.Println(string(jsonData))
}
