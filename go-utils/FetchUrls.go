package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

type UrlResponse struct {
	Urls  []string `json:"urls,omitempty"`
	Error string   `json:"error,omitempty"`
}

func getTopUrls(term string, k int) ([]string, error) {
	baseURL := "https://html.duckduckgo.com/html/"

	// Build URL with query parameters
	params := url.Values{}
	params.Add("q", term)
	fullURL := baseURL + "?" + params.Encode()

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	// Create request with headers
	req, err := http.NewRequest("GET", fullURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

	// Execute request
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch data: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	// Parse HTML
	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse HTML: %v", err)
	}

	// Extract URLs
	urls := []string{}
	doc.Find(".result__a").EachWithBreak(func(i int, s *goquery.Selection) bool {
		if len(urls) >= k {
			return false
		}

		link, exists := s.Attr("href")
		if !exists || link == "" {
			return true
		}

		// Extract actual URL from DuckDuckGo redirect
		if strings.Contains(link, "uddg=") {
			parsedURL, err := url.Parse(link)
			if err == nil {
				queryParams := parsedURL.Query()
				if actualURL := queryParams.Get("uddg"); actualURL != "" {
					link = actualURL
				}
			}
		}

		// Only add valid HTTP/HTTPS URLs
		if strings.HasPrefix(link, "http://") || strings.HasPrefix(link, "https://") {
			urls = append(urls, link)
		}

		return true
	})

	return urls, nil
}

func main() {
	query := flag.String("query", "", "Search query")
	k := flag.Int("k", 5, "Number of URLs to fetch")
	flag.Parse()

	var response UrlResponse

	if *query == "" {
		response.Error = "No query provided"
		jsonOutput, _ := json.Marshal(response)
		fmt.Println(string(jsonOutput))
		return
	}

	urls, err := getTopUrls(*query, *k)
	if err != nil {
		response.Error = fmt.Sprintf("FetchUrls failed: %v", err)
	} else {
		response.Urls = urls
	}

	jsonOutput, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		fmt.Printf("{\"error\": \"JSON marshal error: %v\"}\n", err)
		return
	}

	fmt.Println(string(jsonOutput))
}
