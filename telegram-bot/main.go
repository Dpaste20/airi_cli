package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
	"github.com/joho/godotenv"
)

type Config struct {
	TelegramToken string
	AiriAPIURL    string
	MasterUserID  int64
}

type AiriRequest struct {
	Message   string `json:"message"`
	SessionID string `json:"session_id"`
}

type AiriResponse struct {
	Response string `json:"response"`
}

type ContactData struct {
	ChatID    int64  `json:"chat_id"`
	UserID    int64  `json:"user_id"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
}

var fileMutex sync.Mutex

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
	// log.Printf("🤖 Authorized on account %s", bot.Self.UserName)
	// log.Printf("🔐 Master Access Restricted to User ID: %d", cfg.MasterUserID)
	// log.Printf("🔗 Connected to Airi Backend at: %s", cfg.AiriAPIURL)

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

	masterIDStr := os.Getenv("MASTER_USER_ID")
	if masterIDStr == "" {
		log.Fatal("❌ Error: MASTER_USER_ID is not set in .env file. Bot cannot start without an admin.")
	}

	masterID, err := strconv.ParseInt(masterIDStr, 10, 64)
	if err != nil {
		log.Fatalf("❌ Error: Invalid MASTER_USER_ID format in .env: %v", err)
	}

	return Config{
		TelegramToken: token,
		AiriAPIURL:    apiUrl,
		MasterUserID:  masterID,
	}
}

func handleUpdate(bot *tgbotapi.BotAPI, update tgbotapi.Update, cfg Config) {
	msg := update.Message
	userID := strconv.FormatInt(msg.From.ID, 10)
	chatID := msg.Chat.ID

	saveContactInfo(msg)

	if msg.From.ID != cfg.MasterUserID {
		log.Printf("⛔ Unauthorized access attempt from %s (ID: %d)", msg.From.FirstName, msg.From.ID)

		reply(bot, chatID, "⛔ Access Denied: You are not authorized to interact with this bot.", msg.MessageID)

		return
	}

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
			"Hello Boss %s! 👋\n\n"+
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

func saveContactInfo(msg *tgbotapi.Message) {
	fileMutex.Lock()
	defer fileMutex.Unlock()

	dir := "telegram_contact"
	filePath := filepath.Join(dir, "tg_contact.json")

	if _, err := os.Stat(dir); os.IsNotExist(err) {
		_ = os.Mkdir(dir, 0755)
	}

	var contacts []ContactData
	fileContent, err := os.ReadFile(filePath)
	if err == nil {
		_ = json.Unmarshal(fileContent, &contacts)
	}

	userExists := false
	for _, c := range contacts {
		if c.UserID == msg.From.ID {
			userExists = true
			break
		}
	}

	if !userExists {
		newContact := ContactData{
			ChatID:    msg.Chat.ID,
			UserID:    msg.From.ID,
			FirstName: msg.From.FirstName,
			LastName:  msg.From.LastName,
		}
		contacts = append(contacts, newContact)

		updatedData, err := json.MarshalIndent(contacts, "", "    ")
		if err != nil {
			log.Printf("Error marshalling contact data: %v", err)
			return
		}

		err = os.WriteFile(filePath, updatedData, 0644)
		if err != nil {
			log.Printf("Error writing contact file: %v", err)
		} else {
			log.Printf("📝 New contact saved: %s", msg.From.FirstName)
		}
	}
}
