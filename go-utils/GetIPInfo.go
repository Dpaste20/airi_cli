package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

type IPInfo struct {
	Continent     string  `json:"continent"`
	ContinentCode string  `json:"continentCode"`
	Country       string  `json:"country"`
	CountryCode   string  `json:"countryCode"`
	Region        string  `json:"region"`
	RegionName    string  `json:"regionName"`
	City          string  `json:"city"`
	District      string  `json:"district"`
	Zip           string  `json:"zip"`
	Lat           float64 `json:"lat"`
	Lon           float64 `json:"lon"`
	Timezone      string  `json:"timezone"`
	Offset        int     `json:"offset"`
	Currency      string  `json:"currency"`
	ISP           string  `json:"isp"`
	Org           string  `json:"org"`
	AS            string  `json:"as"`
	ASName        string  `json:"asname"`
	Mobile        bool    `json:"mobile"`
	Proxy         bool    `json:"proxy"`
	Hosting       bool    `json:"hosting"`
	Query         string  `json:"query"`
	Status        string  `json:"status"`
	Message       string  `json:"message,omitempty"`
	DNS           *DNS    `json:"dns,omitempty"`
}

type DNSResponse struct {
	DNS *DNS `json:"dns"`
}

type DNS struct {
	Geo string `json:"geo"`
	IP  string `json:"ip"`
}

func main() {
	client := &http.Client{Timeout: 5 * time.Second}
	var wg sync.WaitGroup

	var ipInfo IPInfo
	var dnsResp DNSResponse
	var ipErr, dnsErr error

	wg.Add(1)
	go func() {
		defer wg.Done()
		resp, err := client.Get("http://ip-api.com/json/?fields=66846719")
		if err != nil {
			ipErr = err
			return
		}
		defer resp.Body.Close()
		if err := json.NewDecoder(resp.Body).Decode(&ipInfo); err != nil {
			ipErr = err
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		resp, err := client.Get("http://edns.ip-api.com/json")
		if err != nil {
			dnsErr = err
			return
		}
		defer resp.Body.Close()
		if err := json.NewDecoder(resp.Body).Decode(&dnsResp); err != nil {
			dnsErr = err
		}
	}()

	wg.Wait()

	if ipErr != nil {
		printError(fmt.Sprintf("Failed to fetch IP info: %v", ipErr))
		return
	}
	if ipInfo.Status == "fail" {
		printError(fmt.Sprintf("API Error: %s", ipInfo.Message))
		return
	}

	if dnsErr == nil && dnsResp.DNS != nil {
		ipInfo.DNS = dnsResp.DNS
	}

	jsonData, err := json.MarshalIndent(ipInfo, "", "    ")
	if err != nil {
		printError(fmt.Sprintf("JSON marshal error: %v", err))
		return
	}

	fmt.Println(string(jsonData))
}

func printError(msg string) {
	fmt.Printf("{\"error\": \"%s\"}\n", msg)
}
