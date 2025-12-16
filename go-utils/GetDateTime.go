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
		CurrentTime: now.Format("15:04:05"),
		// Fomatted date 16/04/2005
		CurrentDate: now.Format("02/01/2006"),
		DayOfWeek:   now.Weekday().String(),
		Timezone:    now.Location().String(),
	}

	jsonData, err := json.Marshal(info)
	if err != nil {
		fmt.Println(`{"error": "Failed to encode time data"}`)
		return
	}
	fmt.Println(string(jsonData))
}
