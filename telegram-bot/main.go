package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
	"github.com/joho/godotenv"
)

type Config struct {
	TelegramToken string
	AiriAPIURL    string
}

type AiriRequest struct {
	Message   string `json:"message"`
	SessionID string `json:"session_id"`
}

type AiriResponse struct {
	Response string `json:"response"`
}

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Println("Info: No .env file found, relying on system environment variables")
	}

	cfg := loadConfig()

	bot, err := tgbotapi.NewBotAPI(cfg.TelegramToken)
	if err != nil {
		log.Fatalf("❌ Error initializing bot: %v", err)
	}

	bot.Debug = false
	log.Printf("🤖 Authorized on account %s", bot.Self.UserName)
	log.Printf("🔗 Connected to Airi Backend at: %s", cfg.AiriAPIURL)

	u := tgbotapi.NewUpdate(0)
	u.Timeout = 60

	updates := bot.GetUpdatesChan(u)

	for update := range updates {

		if update.Message == nil {
			continue
		}

		go handleUpdate(bot, update, cfg)
	}
}

func loadConfig() Config {
	token := os.Getenv("TELEGRAM_TOKEN")
	if token == "" {
		log.Fatal("❌ Error: TELEGRAM_TOKEN environment variable is not set.")
	}

	apiUrl := os.Getenv("AIRI_API_URL")
	if apiUrl == "" {
		apiUrl = "http://localhost:8000/chat"
	}

	return Config{
		TelegramToken: token,
		AiriAPIURL:    apiUrl,
	}
}

func handleUpdate(bot *tgbotapi.BotAPI, update tgbotapi.Update, cfg Config) {
	msg := update.Message
	userID := strconv.FormatInt(msg.From.ID, 10)
	chatID := msg.Chat.ID

	if msg.IsCommand() {
		handleCommand(bot, msg, chatID, userID, cfg)
		return
	}

	if msg.Text != "" {

		sendAction(bot, chatID, tgbotapi.ChatTyping)

		response := sendToAiri(msg.Text, userID, cfg.AiriAPIURL)

		reply(bot, chatID, response, msg.MessageID)
	}
}

func handleCommand(bot *tgbotapi.BotAPI, msg *tgbotapi.Message, chatID int64, userID string, cfg Config) {
	var responseText string
	sendBackendRequest := false
	backendCommand := ""

	switch msg.Command() {
	case "start":
		responseText = fmt.Sprintf(
			"Hello %s! 👋\n\n"+
				"I am connected to Airi's Terminal\n\n"+
				"Available Commands:\n"+
				"🧠 /think - Enable thinking mode\n"+
				"⚡ /fast - Disable thinking mode\n"+
				"ℹ️ /help - Show capabilities",
			msg.From.FirstName,
		)
	case "help":
		sendBackendRequest = true
		backendCommand = "/help"
	case "think":
		sendBackendRequest = true
		backendCommand = "/set think"
	case "fast":
		sendBackendRequest = true
		backendCommand = "/set no_think"
	default:
		responseText = "Unknown command. Try /help."
	}

	if sendBackendRequest {
		sendAction(bot, chatID, tgbotapi.ChatTyping)
		responseText = sendToAiri(backendCommand, userID, cfg.AiriAPIURL)
	}

	reply(bot, chatID, responseText, msg.MessageID)
}

func sendToAiri(message, sessionID, apiURL string) string {
	payload := AiriRequest{
		Message:   message,
		SessionID: "telegram_" + sessionID,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		log.Printf("Error marshalling JSON: %v", err)
		return "❌ Error preparing request."
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post(apiURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Error connecting to Airi: %v", err)
		return "❌ Error: Could not connect to Airi server. Is main.py running?"
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Sprintf("❌ API Error: %d", resp.StatusCode)
	}

	var airiResp AiriResponse
	if err := json.NewDecoder(resp.Body).Decode(&airiResp); err != nil {
		log.Printf("Error decoding response: %v", err)
		return "❌ Error processing response."
	}

	return airiResp.Response
}

// Helper to send "Typing..." status
func sendAction(bot *tgbotapi.BotAPI, chatID int64, action string) {
	msg := tgbotapi.NewChatAction(chatID, action)
	bot.Request(msg)
}

func reply(bot *tgbotapi.BotAPI, chatID int64, text string, replyToID int) {
	msg := tgbotapi.NewMessage(chatID, text)
	msg.ReplyToMessageID = replyToID
	msg.ParseMode = tgbotapi.ModeMarkdown

	if _, err := bot.Send(msg); err != nil {
		log.Printf("Markdown failed, sending plain text. Error: %v", err)
		msg.ParseMode = ""
		bot.Send(msg)
	}
}
