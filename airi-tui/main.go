package main

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"airi-tui/commands"
	toolcommand "airi-tui/tool-command"
	"math/rand"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/gorilla/websocket"
	"gopkg.in/yaml.v3"
)

const airiLogo = `
	   ███████    ███             ███
	 ███▒▒▒▒▒███  ▒▒▒             ▒▒▒
	▒███    ▒███  ████  ████████  ████
	▒███████████ ▒▒███ ▒▒███▒▒███▒▒███
	▒███▒▒▒▒▒███  ▒███  ▒███ ▒▒▒  ▒███
	▒███    ▒███  ▒███  ▒███      ▒███
	█████   █████ █████ █████     █████
	▒▒▒▒▒   ▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒▒     ▒▒▒▒▒
`

var (
	senderStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("5")).Bold(true)
	aiStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("2")).Bold(true)
	errStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Bold(true)
	infoStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Italic(true)
	commandStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("214")).Bold(true)

	recordingStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("196")).
			Background(lipgloss.Color("236")).
			Bold(true).
			Blink(true)

	userBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.ThickBorder(), false, false, false, true).
			BorderForeground(lipgloss.Color("39")).
			Background(lipgloss.Color("236")).
			Padding(1, 2).
			Margin(1, 0, 0, 0)

	aiBoxStyle = lipgloss.NewStyle().
			Padding(0, 3).
			Margin(0, 0, 1, 0)

	helpBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("63")).
			Padding(1, 2).
			Align(lipgloss.Center)

	helpTitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("214")).
			Bold(true).
			Underline(true)

	statusBarStyle = lipgloss.NewStyle().
			Background(lipgloss.Color("236"))

	statusConnectedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("43")).
				Background(lipgloss.Color("237")).
				Padding(0, 1).
				Bold(true)

	statusDisconnectedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("196")).
				Background(lipgloss.Color("52")).
				Padding(0, 1).
				Bold(true)

	statusLabelStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("243")).
				Background(lipgloss.Color("236"))

	statusValueStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("255")).
				Background(lipgloss.Color("236")).
				Bold(true)

	notificationStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("255")).
				Background(lipgloss.Color("63")).
				Bold(true).
				Padding(0, 2)

	acNameStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("255")).
			Background(lipgloss.Color("236")).
			Bold(true)

	acDescStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("243")).
			Background(lipgloss.Color("236"))

	acSelectedRowStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("0")).
				Background(lipgloss.Color("14")).
				Bold(true)

	acSelectedDescStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("0")).
				Background(lipgloss.Color("215"))

	fileBadgeStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("0")).
			Background(lipgloss.Color("214")).
			Bold(true).
			Padding(0, 1)

	fileBarStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("214")).
			Background(lipgloss.Color("236")).
			Padding(0, 1)

	fpOverlayStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("214")).
			Background(lipgloss.Color("232")).
			Padding(1, 2)

	fpTitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("214")).
			Bold(true).
			Underline(true)

	fpDirStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("75")).
			Bold(true)

	fpFileStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("252"))

	fpDimStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240")).
			Italic(true)

	fpSelectedStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("0")).
			Background(lipgloss.Color("214")).
			Bold(true)

	fpHintStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240")).
			Italic(true)
)

type CommandInfo struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

func defaultKnownCommands() []CommandInfo {
	return []CommandInfo{
		{"/help", "Show this screen"},
		{"/attach", "Attach a file (document or image — opens picker, or give a path)"},
		{"/detach", "Remove all staged attachments"},
		{"/save-session", "Save current conversation"},
		{"/resume-session", "Restore a saved conversation"},
		{"/list-sessions", "List all saved sessions"},
		{"/tools-list", "Show all available agent tools"},
	}
}

func loadKnownCommands() []CommandInfo {
	defaultCommands := defaultKnownCommands()

	data, err := os.ReadFile("knownCommands.yaml")
	if err != nil {
		log.Printf("Warning: Failed to read knownCommands.yaml, using defaults: %v", err)
		return defaultCommands
	}

	var structured struct {
		Commands []CommandInfo `yaml:"commands"`
	}

	if err := yaml.Unmarshal(data, &structured); err == nil && len(structured.Commands) > 0 {
		var cleaned []CommandInfo
		for _, c := range structured.Commands {
			if strings.TrimSpace(c.Name) != "" {
				cleaned = append(cleaned, c)
			}
		}
		if len(cleaned) > 0 {
			return cleaned
		}
	}

	var legacy struct {
		Commands []string `yaml:"commands"`
	}

	if err := yaml.Unmarshal(data, &legacy); err != nil {
		log.Printf("Warning: Failed to parse knownCommands.yaml: %v", err)
		return defaultCommands
	}

	if len(legacy.Commands) == 0 {
		return defaultCommands
	}

	var converted []CommandInfo
	for _, name := range legacy.Commands {
		converted = append(converted, CommandInfo{Name: name})
	}
	return converted
}

var knownCommands = loadKnownCommands()

func getMatches(input string) []CommandInfo {
	if !strings.HasPrefix(input, "/") || input == "" {
		return nil
	}
	lower := strings.ToLower(input)
	var matches []CommandInfo
	for _, cmd := range knownCommands {
		if strings.HasPrefix(cmd.Name, lower) && cmd.Name != lower {
			matches = append(matches, cmd)
		}
	}
	return matches
}

type FileAttachment struct {
	Name     string `json:"name"`
	Content  string `json:"content"`
	MimeType string `json:"mime_type"`
}

type AttachedFile struct {
	Name     string
	Path     string
	Content  string
	MimeType string
	Size     int64
}

type ChatRequest struct {
	Message         string           `json:"message,omitempty"`
	SessionID       string           `json:"session_id"`
	SearchKnowledge bool             `json:"search_knowledge"`
	AudioData       string           `json:"audio_data,omitempty"`
	Action          string           `json:"action,omitempty"`
	Files           []FileAttachment `json:"files,omitempty"`
}

type WSMessage struct {
	Type           string  `json:"type"`
	Content        string  `json:"content"`
	Message        string  `json:"message"`
	Error          string  `json:"error"`
	TokenCount     int     `json:"token_count"`
	GenerationTime float64 `json:"generation_time"`
}

type clearNotificationMsg struct{}
type loreTickMsg struct{}

type fpItem struct {
	Name     string
	FullPath string
	IsDir    bool
	IsUp     bool
	Mime     string
	Size     int64
}

type MessageRenderFunc func(m model) string

type model struct {
	conn               *websocket.Conn
	viewport           viewport.Model
	textInput          textinput.Model
	renderer           *glamour.TermRenderer
	messages           []MessageRenderFunc
	currentAiChunk     string
	err                error
	spinner            spinner.Model
	isLoading          bool
	sessionID          string
	connected          bool
	messageCount       int
	startTime          time.Time
	width              int
	height             int
	generatedTokens    int
	generationStart    time.Time
	generationTime     time.Duration
	tokensPerSecond    float64
	isRecording        bool
	recordCmd          *exec.Cmd
	awaitingVoiceChunk bool
	pendingVoiceInput  string
	isSpeaking         bool
	lastAiResponse     string
	notification       string
	showHelp           bool
	suggestions        []CommandInfo
	suggestionIdx      int
	attachedFiles      []AttachedFile

	showFilePicker bool
	fpDir          string
	fpItems        []fpItem
	fpIdx          int

	loreTyping      bool
	loreLogoBlock   string
	loreTypingLines []string
	loreTypingIdx   int
	loreBoxIdx      int
}

func createAiMessageFunc(text string) MessageRenderFunc {
	var cacheWidth int
	var cacheRes string
	return func(m model) string {
		if m.width == cacheWidth && cacheRes != "" {
			return cacheRes
		}
		cacheWidth = m.width
		cacheRes = aiBoxStyle.Width(m.width - 6).MaxWidth(m.width - 6).Render(m.renderMarkdown(text))
		return cacheRes
	}
}

func createUserMessageFunc(text string) MessageRenderFunc {
	var cacheWidth int
	var cacheRes string
	return func(m model) string {
		if m.width == cacheWidth && cacheRes != "" {
			return cacheRes
		}
		cacheWidth = m.width
		cacheRes = userBoxStyle.Width(m.width).MaxWidth(m.width).Render(text)
		return cacheRes
	}
}

func createSystemMessageFunc(text string) MessageRenderFunc {
	var cacheWidth int
	var cacheRes string
	return func(m model) string {
		if m.width == cacheWidth && cacheRes != "" {
			return cacheRes
		}
		cacheWidth = m.width
		cacheRes = infoStyle.Render(text)
		return cacheRes
	}
}

func createErrorMessageFunc(text string) MessageRenderFunc {
	var cacheWidth int
	var cacheRes string
	return func(m model) string {
		if m.width == cacheWidth && cacheRes != "" {
			return cacheRes
		}
		cacheWidth = m.width
		cacheRes = aiBoxStyle.BorderForeground(lipgloss.Color("9")).Width(m.width - 6).MaxWidth(m.width - 6).Render(errStyle.Render(text))
		return cacheRes
	}
}

func (m model) updateViewportContent() model {
	var lines []string
	for _, fn := range m.messages {
		lines = append(lines, fn(m))
	}
	content := strings.Join(lines, "\n")
	if m.currentAiChunk != "" {
		renderedChunk := m.renderMarkdown(m.currentAiChunk)
		boxedContent := aiBoxStyle.Width(m.width - 6).MaxWidth(m.width - 6).Render(renderedChunk)
		content += "\n" + boxedContent
	} else if m.isLoading {
		content += "\n" + m.spinner.View()
	}
	m.viewport.SetContent(content)
	return m
}

func (m model) getRenderedMessages() []string {
	var res []string
	for _, fn := range m.messages {
		res = append(res, fn(m))
	}
	return res
}

const maxSuggestionRows = 8

func visibleSuggestionCount(n int) int {
	if n > maxSuggestionRows {
		return maxSuggestionRows
	}
	return n
}

func (m model) syncLayout() model {
	if m.height == 0 {
		return m
	}

	offset := 7
	if len(m.attachedFiles) > 0 {
		offset += 2
	}
	if len(m.suggestions) > 0 {
		offset += visibleSuggestionCount(len(m.suggestions)) + 1
	}

	newHeight := m.height - offset
	if newHeight < 1 {
		newHeight = 1
	}

	if m.viewport.Height != newHeight {
		m.viewport.Height = newHeight
		m.viewport.GotoBottom()
	}
	return m
}

func mimeFromExt(path string) (string, bool) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".txt", ".md", ".log":
		return "text/plain", true
	case ".csv":
		return "text/csv", true
	case ".pdf":
		return "application/pdf", true
	case ".doc":
		return "application/msword", true
	case ".docx", ".docm":
		return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", true
	case ".ppt", ".pps", ".pot":
		return "application/vnd.ms-powerpoint", true
	case ".pptx", ".pptm", ".ppsx", ".ppsm":
		return "application/vnd.openxmlformats-officedocument.presentationml.presentation", true
	case ".xls":
		return "application/vnd.ms-excel", true
	case ".xlsx", ".xlsm", ".xlsb":
		return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", true
	case ".odt":
		return "application/vnd.oasis.opendocument.text", true
	case ".ods":
		return "application/vnd.oasis.opendocument.spreadsheet", true
	case ".odp":
		return "application/vnd.oasis.opendocument.presentation", true
	case ".rtf":
		return "application/rtf", true
	case ".epub":
		return "application/epub+zip", true
	case ".png":
		return "image/png", true
	case ".jpg", ".jpeg":
		return "image/jpeg", true
	case ".gif":
		return "image/gif", true
	case ".bmp":
		return "image/bmp", true
	case ".webp":
		return "image/webp", true
	case ".svg":
		return "image/svg+xml", true
	case ".tiff", ".tif":
		return "image/tiff", true
	case ".ico":
		return "image/x-icon", true
	case ".avif":
		return "image/avif", true
	case ".heic", ".heif":
		return "image/heic", true
	default:
		return "", false
	}
}

func expandHome(path string) string {
	if strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}

func loadAttachedFile(rawPath string) (AttachedFile, error) {
	path := expandHome(strings.TrimSpace(rawPath))

	mime, ok := mimeFromExt(path)
	if !ok {
		return AttachedFile{}, fmt.Errorf("unsupported file type — attach a document (.docx, .pptx, .xlsx, .pdf, ...) or image (.png, .jpg, .webp, ...)")
	}

	info, err := os.Stat(path)
	if err != nil {
		return AttachedFile{}, fmt.Errorf("file not found: %s", path)
	}
	maxBytes := int64(20 * 1024 * 1024)
	if strings.HasPrefix(mime, "image/") {
		maxBytes = int64(15 * 1024 * 1024)
	}
	if info.Size() > maxBytes {
		return AttachedFile{}, fmt.Errorf("file too large (max %d MB): %s", maxBytes/1024/1024, filepath.Base(path))
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return AttachedFile{}, fmt.Errorf("could not read file: %v", err)
	}

	return AttachedFile{
		Name:     filepath.Base(path),
		Path:     path,
		Content:  base64.StdEncoding.EncodeToString(data),
		MimeType: mime,
		Size:     info.Size(),
	}, nil
}

func loadFpDir(dir string) ([]fpItem, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var dirs, files []fpItem

	for _, e := range entries {

		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		fullPath := filepath.Join(dir, e.Name())
		if e.IsDir() {
			dirs = append(dirs, fpItem{
				Name:     e.Name() + "/",
				FullPath: fullPath,
				IsDir:    true,
			})
		} else {
			info, _ := e.Info()
			size := int64(0)
			if info != nil {
				size = info.Size()
			}
			mime, _ := mimeFromExt(e.Name())
			files = append(files, fpItem{
				Name:     e.Name(),
				FullPath: fullPath,
				Mime:     mime,
				Size:     size,
			})
		}
	}

	sort.Slice(dirs, func(i, j int) bool { return dirs[i].Name < dirs[j].Name })
	sort.Slice(files, func(i, j int) bool { return files[i].Name < files[j].Name })

	var items []fpItem

	if filepath.Dir(dir) != dir {
		items = append(items, fpItem{
			Name:     "../",
			FullPath: filepath.Dir(dir),
			IsDir:    true,
			IsUp:     true,
		})
	}
	items = append(items, dirs...)
	items = append(items, files...)
	return items, nil
}

func (m model) renderFilePicker() string {
	const maxVisible = 16

	title := fpTitleStyle.Render("📂 File Picker")
	path := fpHintStyle.Render(" " + m.fpDir)

	start := 0
	if m.fpIdx >= maxVisible {
		start = m.fpIdx - maxVisible + 1
	}
	end := start + maxVisible
	if end > len(m.fpItems) {
		end = len(m.fpItems)
	}

	var rows []string
	for i := start; i < end; i++ {
		item := m.fpItems[i]

		var label string
		switch {
		case item.IsUp:
			label = fpDirStyle.Render("  ↑  ../")
		case item.IsDir:
			label = fpDirStyle.Render("  📁 " + item.Name)
		case item.Mime != "":
			icon := "📄 "
			if item.Mime == "application/pdf" {
				icon = "📕 "
			} else if strings.HasPrefix(item.Mime, "image/") {
				icon = "🖼️ "
			}
			sizeStr := fmt.Sprintf("%.1f KB", float64(item.Size)/1024)
			if item.Size >= 1024*1024 {
				sizeStr = fmt.Sprintf("%.1f MB", float64(item.Size)/1024/1024)
			}
			label = fpFileStyle.Render(fmt.Sprintf("  %s%-36s %s", icon, item.Name, sizeStr))
		default:
			label = fpDimStyle.Render("  ·  " + item.Name)
		}

		if i == m.fpIdx {

			plain := item.Name
			if item.IsUp {
				plain = "../"
			}
			_ = plain
			label = fpSelectedStyle.Render(label)
		}

		rows = append(rows, label)
	}

	scrollInfo := ""
	if len(m.fpItems) > maxVisible {
		scrollInfo = fpHintStyle.Render(fmt.Sprintf(" %d/%d", m.fpIdx+1, len(m.fpItems)))
	}

	empty := ""
	if len(m.fpItems) == 0 {
		empty = fpDimStyle.Render("  (empty directory)")
	}

	hints := fpHintStyle.Render("↑↓ navigate  Enter select/open  Backspace parent  Esc close")

	innerWidth := m.width - 8
	if innerWidth < 40 {
		innerWidth = 40
	}

	body := fmt.Sprintf("%s%s\n%s\n\n%s%s\n\n%s",
		title, scrollInfo,
		path,
		strings.Join(rows, "\n"),
		empty,
		hints,
	)

	box := fpOverlayStyle.Width(innerWidth).MaxWidth(innerWidth).Render(body)

	return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, box)
}

func newSessionID() string {
	const chars = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		b[i] = chars[rand.Intn(len(chars))]
	}
	return "terminal-" + string(b)
}

func loadWelcomeMessages() []string {
	defaultMessages := []string{
		"Connected to Airi Terminal. Awaiting your command.",
		"Terminal initialized. Let's get to work.",
	}

	data, err := os.ReadFile("welcome_msg.yaml")
	if err != nil {
		return defaultMessages
	}

	var config struct {
		WelcomeMessages []string `yaml:"welcome_messages"`
	}

	if err := yaml.Unmarshal(data, &config); err != nil {
		log.Printf("Warning: Failed to parse welcome_msg.yaml: %v", err)
		return defaultMessages
	}

	if len(config.WelcomeMessages) == 0 {
		return defaultMessages
	}

	return config.WelcomeMessages
}

func initialModel(conn *websocket.Conn) model {
	ti := textinput.New()
	ti.Placeholder = "Ask Airi something... (type /help for commands)"
	ti.Prompt = ""
	ti.Focus()
	ti.CharLimit = 1000
	ti.Width = 20

	boxBg := lipgloss.Color("236")
	ti.TextStyle = lipgloss.NewStyle().Background(boxBg)
	ti.PlaceholderStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Background(boxBg)
	ti.Cursor.Style = lipgloss.NewStyle().Background(boxBg).Foreground(lipgloss.Color("255"))

	defaultWidth := 80
	vp := viewport.New(defaultWidth, 20)

	welcomeMessages := loadWelcomeMessages()
	selectedWelcome := welcomeMessages[rand.Intn(len(welcomeMessages))]
	go func() {
		time.Sleep(1 * time.Second)
		spokenText := ", , , " + selectedWelcome
		_ = exec.Command("spd-say", spokenText).Run()
	}()

	renderer, _ := glamour.NewTermRenderer(
		glamour.WithStandardStyle("dark"),
		glamour.WithWordWrap(74),
	)
	s := spinner.New()
	s.Spinner = spinner.MiniDot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("cyan"))

	m := model{
		conn:      conn,
		textInput: ti,
		viewport:  vp,
		renderer:  renderer,
		messages: []MessageRenderFunc{
			func(m model) string {
				coloredLogo := aiStyle.Render(airiLogo)
				return lipgloss.PlaceHorizontal(m.width, lipgloss.Center, coloredLogo)
			},
			createSystemMessageFunc("\n" + selectedWelcome + "\n"),
		},
		spinner:            s,
		isLoading:          false,
		sessionID:          newSessionID(),
		connected:          true,
		messageCount:       0,
		startTime:          time.Now(),
		width:              defaultWidth,
		height:             24,
		generatedTokens:    0,
		generationTime:     0,
		tokensPerSecond:    0.0,
		isRecording:        false,
		awaitingVoiceChunk: false,
		pendingVoiceInput:  "",
		isSpeaking:         false,
		lastAiResponse:     "",
		notification:       "",
		showHelp:           false,
		suggestions:        nil,
		suggestionIdx:      -1,
		attachedFiles:      nil,
		showFilePicker:     false,
		fpDir:              "",
		fpItems:            nil,
		fpIdx:              0,
	}

	m = m.updateViewportContent()
	return m
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		textinput.Blink,
		waitForIncomingMessage(m.conn),
	)
}

func preprocessMarkdown(text string) string {
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		if strings.HasPrefix(line, "### ") {
			lines[i] = "**" + strings.TrimPrefix(line, "### ") + "**"
		} else if strings.HasPrefix(line, "## ") {
			lines[i] = "**" + strings.TrimPrefix(line, "## ") + "**"
		}
	}
	return strings.Join(lines, "\n")
}

func (m model) renderMarkdown(text string) string {
	text = preprocessMarkdown(text)
	tr, err := m.renderer.Render(text)
	if err != nil {
		return text
	}
	return tr
}

func (m model) renderStatusBar() string {
	if m.notification != "" {
		msg := notificationStyle.Render(m.notification)
		return statusBarStyle.Render(msg)
	}

	var connStatus string
	if m.connected {
		connStatus = statusConnectedStyle.Render("● Connected")
	} else {
		connStatus = statusDisconnectedStyle.Render("● Disconnected")
	}

	var recStatus string
	if m.isRecording {
		recStatus = recordingStyle.Render("● Listening")
	} else if m.isSpeaking {
		recStatus = statusLabelStyle.Render("Speaking")
	} else {
		recStatus = statusLabelStyle.Render("Space to Talk")
	}

	msgCount := statusLabelStyle.Render("Messages: ") + statusValueStyle.Render(fmt.Sprintf("%d", m.messageCount))
	sessionInfo := statusLabelStyle.Render("Session: ") + statusValueStyle.Render(m.sessionID)
	genTokens := statusLabelStyle.Render("Tokens: ") + statusValueStyle.Render(fmt.Sprintf("%d", m.generatedTokens))
	genTime := statusLabelStyle.Render("Time: ") + statusValueStyle.Render(fmt.Sprintf("%.2fs", m.generationTime.Seconds()))
	genSpeed := statusLabelStyle.Render("Speed: ") + statusValueStyle.Render(fmt.Sprintf("%.2f t/s", m.tokensPerSecond))

	spacer := lipgloss.NewStyle().Background(lipgloss.Color("236")).Render("  ")

	statusContent := lipgloss.JoinHorizontal(lipgloss.Left,
		connStatus,
		spacer,
		recStatus,
		spacer,
		msgCount,
		spacer,
		sessionInfo,
		spacer,
		genTokens,
		spacer,
		genTime,
		spacer,
		genSpeed,
	)

	maxWidth := m.width - 6
	if maxWidth < 1 {
		maxWidth = 1
	}

	return statusBarStyle.MaxWidth(maxWidth).Render(statusContent)
}

func (m model) openFilePicker() (model, tea.Cmd) {
	startDir, err := os.UserHomeDir()
	if err != nil {
		startDir, _ = os.Getwd()
	}
	items, err := loadFpDir(startDir)
	if err != nil {
		m.notification = "✗ Cannot open " + startDir
		return m, tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
			return clearNotificationMsg{}
		})
	}
	m.fpDir = startDir
	m.fpItems = items
	m.fpIdx = 0
	m.showFilePicker = true
	return m, nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
	)

	switch msg := msg.(type) {

	case tea.KeyMsg:
		if m.showFilePicker {
			switch msg.Type {
			case tea.KeyCtrlC:
				return m, tea.Quit

			case tea.KeyEsc, tea.KeyCtrlF:
				m.showFilePicker = false
				return m, nil

			case tea.KeyUp:
				if m.fpIdx > 0 {
					m.fpIdx--
				}
				return m, nil

			case tea.KeyDown:
				if m.fpIdx < len(m.fpItems)-1 {
					m.fpIdx++
				}
				return m, nil

			case tea.KeyBackspace:
				parent := filepath.Dir(m.fpDir)
				if parent != m.fpDir {
					items, err := loadFpDir(parent)
					if err == nil {
						m.fpDir = parent
						m.fpItems = items
						m.fpIdx = 0
					}
				}
				return m, nil

			case tea.KeyEnter:
				if len(m.fpItems) == 0 {
					return m, nil
				}
				item := m.fpItems[m.fpIdx]

				if item.IsDir {
					items, err := loadFpDir(item.FullPath)
					if err != nil {
						m.notification = "✗ " + err.Error()
						return m, tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
							return clearNotificationMsg{}
						})
					}
					m.fpDir = item.FullPath
					m.fpItems = items
					m.fpIdx = 0
					return m, nil
				}

				if item.Mime == "" {
					m.notification = "⚠ Unsupported file type — attach a document (.docx, .pptx, .xlsx, .pdf, ...) or image (.png, .jpg, .webp, ...)"
					return m, tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
						return clearNotificationMsg{}
					})
				}

				af, err := loadAttachedFile(item.FullPath)
				if err != nil {
					m.notification = "✗ " + err.Error()
					return m, tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
						return clearNotificationMsg{}
					})
				}

				for _, existing := range m.attachedFiles {
					if existing.Path == af.Path {
						m.notification = "⚠ Already attached: " + af.Name
						m.showFilePicker = false
						return m, tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
							return clearNotificationMsg{}
						})
					}
				}

				m.attachedFiles = append(m.attachedFiles, af)
				m.showFilePicker = false

				sizeStr := fmt.Sprintf("%.1f KB", float64(af.Size)/1024)
				if af.Size >= 1024*1024 {
					sizeStr = fmt.Sprintf("%.1f MB", float64(af.Size)/1024/1024)
				}
				m.notification = fmt.Sprintf("📎 %s (%s) attached", af.Name, sizeStr)
				m = m.syncLayout()
				return m, tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
					return clearNotificationMsg{}
				})
			}
			return m, nil
		}

		if m.showHelp {
			if msg.Type == tea.KeyCtrlX {
				m.showHelp = false
			}

			if msg.Type == tea.KeyCtrlC {
				return m, tea.Quit
			}
			return m, nil
		}

		switch msg.Type {
		case tea.KeyCtrlC:
			if m.isRecording && m.recordCmd != nil {
				_ = m.recordCmd.Process.Kill()
			}
			return m, tea.Quit

		case tea.KeyCtrlY:
			if m.lastAiResponse != "" {
				err := clipboard.WriteAll(m.lastAiResponse)
				if err == nil {
					m.notification = "✓ Copied to clipboard"
				} else {
					m.notification = "✗ Copy failed"
				}

				return m, tea.Tick(time.Millisecond*500, func(_ time.Time) tea.Msg {
					return clearNotificationMsg{}
				})
			}
			return m, nil

		case tea.KeyCtrlD:
			if len(m.attachedFiles) > 0 {
				m.attachedFiles = nil
				m.notification = "✓ Attachments cleared"
				m = m.syncLayout()
				return m, tea.Tick(time.Millisecond*800, func(_ time.Time) tea.Msg {
					return clearNotificationMsg{}
				})
			}
			return m, nil

		case tea.KeyCtrlF:
			if m.showFilePicker {
				m.showFilePicker = false
				return m, nil
			}
			return m.openFilePicker()

		case tea.KeyTab:
			if len(m.suggestions) > 0 {
				idx := m.suggestionIdx
				if idx < 0 {
					idx = 0
				}
				m.textInput.SetValue(m.suggestions[idx].Name)
				m.textInput.CursorEnd()
				m.suggestions = nil
				m.suggestionIdx = -1
				m = m.syncLayout()
			}
			return m, nil

		case tea.KeyUp:
			if len(m.suggestions) > 0 {
				m.suggestionIdx--
				if m.suggestionIdx < 0 {
					m.suggestionIdx = len(m.suggestions) - 1
				}
				return m, nil
			}

		case tea.KeyDown:
			if len(m.suggestions) > 0 {
				m.suggestionIdx++
				if m.suggestionIdx >= len(m.suggestions) {
					m.suggestionIdx = 0
				}
				return m, nil
			}

		case tea.KeyEsc:
			if len(m.suggestions) > 0 {
				m.suggestions = nil
				m.suggestionIdx = -1
				m = m.syncLayout()
				return m, nil
			}
			if m.isRecording && m.recordCmd != nil {
				_ = m.recordCmd.Process.Kill()
			}
			return m, tea.Quit

		case tea.KeySpace:

			if m.textInput.Value() == "" {

				if m.isRecording {
					m.isRecording = false
					if m.recordCmd != nil && m.recordCmd.Process != nil {
						_ = m.recordCmd.Process.Signal(os.Interrupt)
						time.Sleep(100 * time.Millisecond)
					}

					audioBytes, err := os.ReadFile("/tmp/airi_voice.wav")
					if err != nil {
						m.messages = append(m.messages, createErrorMessageFunc("Error reading audio: "+err.Error()))
						m = m.updateViewportContent()
						m.viewport.GotoBottom()
						return m, nil
					}

					encodedAudio := base64.StdEncoding.EncodeToString(audioBytes)

					m.awaitingVoiceChunk = true
					m.isLoading = true
					m.generatedTokens = 0
					m.generationTime = 0
					m.tokensPerSecond = 0.0

					req := ChatRequest{
						SessionID: m.sessionID,
						AudioData: encodedAudio,
					}

					sendCmd := func() tea.Msg {
						err := m.conn.WriteJSON(req)
						if err != nil {
							m.connected = false
							return err
						}
						return nil
					}
					return m, tea.Batch(sendCmd, m.spinner.Tick)
				}

				if m.isSpeaking {
					stopSpeechCmd := func() tea.Msg {
						req := ChatRequest{
							SessionID: m.sessionID,
							Action:    "stop_speech",
						}
						_ = m.conn.WriteJSON(req)
						return nil
					}
					stopSpeechCmd()
					m.isSpeaking = false
				}

				m.isRecording = true
				m.recordCmd = exec.Command("arecord", "-f", "cd", "/tmp/airi_voice.wav")
				if err := m.recordCmd.Start(); err != nil {
					m.messages = append(m.messages, createErrorMessageFunc("Error starting recording: "+err.Error()))
					m = m.updateViewportContent()
					m.viewport.GotoBottom()
					m.isRecording = false
					return m, nil
				}
				return m, m.spinner.Tick
			}

		case tea.KeyEnter:

			if len(m.suggestions) > 0 {
				idx := m.suggestionIdx
				if idx < 0 {
					idx = 0
				}
				m.textInput.SetValue(m.suggestions[idx].Name)
				m.textInput.CursorEnd()
				m.suggestions = nil
				m.suggestionIdx = -1
				m = m.syncLayout()
				return m, nil
			}
			m.suggestions = nil
			m.suggestionIdx = -1

			input := m.textInput.Value()
			if input == "" {
				return m, nil
			}

			if input == "/help" {
				m.showHelp = true
				m.textInput.SetValue("")
				return m, nil
			}

			if strings.HasPrefix(input, "/attach") {
				m.textInput.SetValue("")
				rawPath := strings.TrimSpace(strings.TrimPrefix(input, "/attach"))

				if rawPath == "" {
					return m.openFilePicker()
				}

				m.messages = append(m.messages, createUserMessageFunc(input))
				m.messageCount++

				af, err := loadAttachedFile(rawPath)
				if err != nil {
					m.messages = append(m.messages, createErrorMessageFunc("❌ "+err.Error()))
					m.messageCount++
					m = m.updateViewportContent()
					m.viewport.GotoBottom()
					m = m.syncLayout()
					return m, nil
				}

				for _, existing := range m.attachedFiles {
					if existing.Path == af.Path {
						m.notification = "⚠ Already attached: " + af.Name
						return m, tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
							return clearNotificationMsg{}
						})
					}
				}

				m.attachedFiles = append(m.attachedFiles, af)

				sizeStr := fmt.Sprintf("%.1f KB", float64(af.Size)/1024)
				if af.Size >= 1024*1024 {
					sizeStr = fmt.Sprintf("%.1f MB", float64(af.Size)/1024/1024)
				}
				okMsg := fmt.Sprintf("📎 Attached **%s** (%s)  —  will be sent with your next message.", af.Name, sizeStr)
				m.messages = append(m.messages, createAiMessageFunc(okMsg))
				m.messageCount++
				m = m.updateViewportContent()
				m.viewport.GotoBottom()
				m = m.syncLayout()
				return m, nil
			}

			if input == "/detach" {
				m.textInput.SetValue("")
				m.messages = append(m.messages, createUserMessageFunc(input))
				m.messageCount++

				if len(m.attachedFiles) == 0 {
					m.messages = append(m.messages, func(m model) string {
						return aiBoxStyle.Width(m.width - 6).MaxWidth(m.width - 6).Render(infoStyle.Render("No files are currently attached."))
					})
				} else {
					m.attachedFiles = nil
					m.messages = append(m.messages, createAiMessageFunc("🗑 All attachments cleared."))
				}
				m.messageCount++
				m = m.updateViewportContent()
				m.viewport.GotoBottom()
				m = m.syncLayout()
				return m, nil
			}

			if result, handled := commands.Dispatch(input, m.sessionID, m.getRenderedMessages()); handled {
				m.textInput.SetValue("")

				m.messages = append(m.messages, createUserMessageFunc(input))
				m.messageCount++

				if result.ViewportMessage != "" {
					if result.IsError {
						m.messages = append(m.messages, createErrorMessageFunc(result.ViewportMessage))
					} else {
						m.messages = append(m.messages, createAiMessageFunc(result.ViewportMessage))
					}
					m.messageCount++
				}

				var notifCmd tea.Cmd
				if len(result.TypingLines) > 0 {
					m.loreLogoBlock = result.LogoBlock
					m.loreTypingLines = result.TypingLines
					m.loreTypingIdx = 0
					m.loreTyping = true

					m.messages = append(m.messages, func(m model) string { return "" })
					m.loreBoxIdx = len(m.messages) - 1
					m.messageCount++
					notifCmd = tea.Tick(80*time.Millisecond, func(_ time.Time) tea.Msg {
						return loreTickMsg{}
					})
				} else if result.Notification != "" {
					m.notification = result.Notification
					notifCmd = tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
						return clearNotificationMsg{}
					})
				}

				if result.NewSessionID != "" {
					m.sessionID = result.NewSessionID
				}
				if result.RestoredMessages != nil {
					m.messages = nil
					for _, rm := range result.RestoredMessages {
						text := rm
						m.messages = append(m.messages, func(m model) string {
							return lipgloss.NewStyle().MaxWidth(m.width).Render(text)
						})
					}
					m.messageCount = len(result.RestoredMessages)
				}

				m = m.updateViewportContent()
				m.viewport.GotoBottom()
				return m, notifCmd
			}

			if m.isSpeaking {
				stopSpeechCmd := func() tea.Msg {
					req := ChatRequest{
						SessionID: m.sessionID,
						Action:    "stop_speech",
					}
					_ = m.conn.WriteJSON(req)
					return nil
				}
				stopSpeechCmd()
				m.isSpeaking = false
			}

			userBubbleText := input
			if len(m.attachedFiles) > 0 {
				var names []string
				for _, af := range m.attachedFiles {
					names = append(names, "📎 "+af.Name)
				}
				userBubbleText += "\n" + infoStyle.Render(strings.Join(names, "  "))
			}

			m.messages = append(m.messages, createUserMessageFunc(userBubbleText))
			m.messageCount++

			inputTrimmed := strings.TrimSpace(input)
			parts := strings.SplitN(inputTrimmed, " ", 2)
			cmdTrigger := parts[0]
			cmdArgs := ""
			if len(parts) > 1 {
				cmdArgs = strings.TrimSpace(parts[1])
			}

			actualPrompt := input
			if macroPrompt, exists := toolcommand.CommandMacros[cmdTrigger]; exists {

				if strings.Contains(macroPrompt, "%s") {
					actualPrompt = fmt.Sprintf(macroPrompt, cmdArgs)
				} else {
					actualPrompt = macroPrompt
				}
			}

			m.currentAiChunk = ""
			m.isLoading = true

			m.generatedTokens = 0
			m.generationTime = 0
			m.tokensPerSecond = 0.0

			m = m.updateViewportContent()
			m.viewport.GotoBottom()

			var wireFiles []FileAttachment
			for _, af := range m.attachedFiles {
				wireFiles = append(wireFiles, FileAttachment{
					Name:     af.Name,
					Content:  af.Content,
					MimeType: af.MimeType,
				})
			}
			m.attachedFiles = nil

			req := ChatRequest{
				Message:         actualPrompt,
				SessionID:       m.sessionID,
				SearchKnowledge: false,
				Files:           wireFiles,
			}

			sendCmd := func() tea.Msg {
				err := m.conn.WriteJSON(req)
				if err != nil {
					m.connected = false
					return err
				}
				return nil
			}

			m.textInput.SetValue("")
			m = m.syncLayout()
			return m, tea.Batch(sendCmd, m.spinner.Tick)
		}

	case loreTickMsg:
		if m.loreTyping && m.loreTypingIdx < len(m.loreTypingLines) {
			m.loreTypingIdx++
			typedBlock := lipgloss.NewStyle().PaddingLeft(4).PaddingTop(2).
				Render(strings.Join(m.loreTypingLines[:m.loreTypingIdx], "\n"))
			combined := lipgloss.JoinHorizontal(lipgloss.Top, m.loreLogoBlock, typedBlock)

			m.messages[m.loreBoxIdx] = func(m model) string {
				return aiBoxStyle.Width(m.width - 6).MaxWidth(m.width - 6).Render(combined)
			}
			m = m.updateViewportContent()
			m.viewport.GotoBottom()

			if m.loreTypingIdx < len(m.loreTypingLines) {
				return m, tea.Tick(80*time.Millisecond, func(_ time.Time) tea.Msg {
					return loreTickMsg{}
				})
			}
			m.loreTyping = false
			m.notification = "Lore loaded"
			return m, tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
				return clearNotificationMsg{}
			})
		}

	case clearNotificationMsg:
		m.notification = ""

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		m.viewport.Width = msg.Width
		m.textInput.Width = msg.Width - 6

		m.renderer, _ = glamour.NewTermRenderer(
			glamour.WithStandardStyle("dark"),
			glamour.WithWordWrap(msg.Width-8),
		)

		m = m.updateViewportContent()
		m = m.syncLayout()

	case spinner.TickMsg:
		if m.isLoading {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)

			if m.currentAiChunk == "" {
				m = m.updateViewportContent()
				m.viewport.GotoBottom()
			}

			return m, cmd
		} else if m.isRecording {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			return m, cmd
		}

	case WSMessage:
		switch msg.Type {
		case "start":
			m.isLoading = true
			m.generationStart = time.Now()
			m.generatedTokens = 0

		case "chunk":

			if m.awaitingVoiceChunk && strings.Contains(msg.Content, "🎤 *Voice:*") {

				voiceText := strings.TrimPrefix(msg.Content, "🎤 *Voice:* ")
				voiceText = strings.TrimSpace(voiceText)

				m.messages = append(m.messages, createUserMessageFunc(voiceText))
				m.messageCount++
				m.pendingVoiceInput = voiceText
				m.awaitingVoiceChunk = false

				m = m.updateViewportContent()
				m.viewport.GotoBottom()
			} else {

				m.currentAiChunk += msg.Content

				m.generatedTokens = len(m.currentAiChunk) / 4
				elapsed := time.Since(m.generationStart).Seconds()
				if elapsed > 0 {
					m.tokensPerSecond = float64(m.generatedTokens) / elapsed
				}

				m = m.updateViewportContent()
				m.viewport.GotoBottom()
			}

		case "end":
			if msg.GenerationTime > 0 {
				m.generationTime = time.Duration(msg.GenerationTime * float64(time.Second))
			} else {
				m.generationTime = time.Since(m.generationStart)
			}

			if msg.TokenCount > 0 {
				m.generatedTokens = msg.TokenCount
			} else {
				m.generatedTokens = len(m.currentAiChunk) / 4
			}

			if m.generationTime.Seconds() > 0 {
				m.tokensPerSecond = float64(m.generatedTokens) / m.generationTime.Seconds()
			}

			m.lastAiResponse = m.currentAiChunk

			m.messages = append(m.messages, createAiMessageFunc(m.currentAiChunk))
			m.messageCount++
			m.currentAiChunk = ""
			m.isLoading = false
			m.pendingVoiceInput = ""
			m.isSpeaking = true

		case "speech_stopped":
			m.isSpeaking = false

		case "error":
			m.messages = append(m.messages, createErrorMessageFunc("Error: "+msg.Message))
			m = m.updateViewportContent()
			m.viewport.GotoBottom()
			m.isLoading = false
			m.isSpeaking = false

		default:
			if msg.Error != "" {
				m.messages = append(m.messages, createErrorMessageFunc("System Error: "+msg.Error))
				m = m.updateViewportContent()
				m.viewport.GotoBottom()
				m.isLoading = false
				m.connected = false
				m.isSpeaking = false
			}
		}

		return m, waitForIncomingMessage(m.conn)

	case error:
		m.err = msg
		m.connected = false
		m.isSpeaking = false
		return m, nil
	}

	m.textInput, tiCmd = m.textInput.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)

	newMatches := getMatches(m.textInput.Value())
	if len(newMatches) != len(m.suggestions) {
		m.suggestionIdx = -1
	}
	m.suggestions = newMatches

	m = m.syncLayout()

	return m, tea.Batch(tiCmd, vpCmd)
}

func (m model) renderFileBar() string {
	if len(m.attachedFiles) == 0 {
		return ""
	}
	var badges []string
	for _, af := range m.attachedFiles {
		icon := "📄"
		if af.MimeType == "application/pdf" {
			icon = "📕"
		} else if strings.HasPrefix(af.MimeType, "image/") {
			icon = "🖼️"
		}
		badges = append(badges, fileBadgeStyle.Render(icon+" "+af.Name))
	}
	hint := fileBarStyle.Render("ctrl+d to clear")
	return strings.Join(badges, " ") + "  " + hint
}

func (m model) View() string {
	if m.err != nil {
		return errStyle.Render(fmt.Sprintf("\nFatal Error: %v\nRestart the application.", m.err))
	}

	if m.showFilePicker {
		return m.renderFilePicker()
	}

	if m.showHelp {
		title := helpTitleStyle.Render("Commands / Short Cuts :")

		content := fmt.Sprintf(`
%s

%s
  /attach [path]          Attach a file (document or image — opens picker if no path)
  /detach                 Remove all staged attachments
  /save-session [name]    Save current conversation
  /resume-session [name]  Restore a saved conversation
  /resume-session         List all saved sessions
  /tools-list             Show all available agent tools
  /tools-list [keyword]   Filter tools by name or category
  /help                   Show this screen

%s
  ctrl + f   Open file picker (browse & attach)
  ctrl + y   Copy last Airi response to clipboard
  ctrl + d   Clear all staged file attachments
  ctrl + c   Quit

%s
`, title,
			commandStyle.Render("Slash Commands:"),
			commandStyle.Render("Shortcuts:"),
			infoStyle.Render("Press ctrl + x to close"))

		helpBox := helpBoxStyle.Render(content)

		return lipgloss.Place(
			m.width,
			m.height,
			lipgloss.Center,
			lipgloss.Center,
			helpBox,
		)
	}

	statusBar := m.renderStatusBar()
	innerContent := statusBar

	if len(m.suggestions) > 0 {
		rowWidth := m.width - 6
		if rowWidth < 10 {
			rowWidth = 10
		}

		total := len(m.suggestions)
		visibleCount := visibleSuggestionCount(total)

		selected := m.suggestionIdx
		if selected < 0 {
			selected = 0
		}

		windowStart := 0
		if total > visibleCount {
			windowStart = selected - visibleCount + 1
			if windowStart < 0 {
				windowStart = 0
			}
			if windowStart > total-visibleCount {
				windowStart = total - visibleCount
			}
		}
		visible := m.suggestions[windowStart : windowStart+visibleCount]
		selectedInWindow := selected - windowStart

		nameWidth := 0
		for _, s := range visible {
			if l := len(s.Name); l > nameWidth {
				nameWidth = l
			}
		}
		nameWidth += 3

		var rows []string
		for i, cmd := range visible {
			name := cmd.Name
			pad := nameWidth - len(name)
			if pad < 1 {
				pad = 1
			}
			line := "  " + name + strings.Repeat(" ", pad) + cmd.Description

			if i == selectedInWindow {
				rows = append(rows, acSelectedRowStyle.Width(rowWidth).MaxWidth(rowWidth).Render(line))
			} else {
				row := acNameStyle.Render("  "+name+strings.Repeat(" ", pad)) + acDescStyle.Render(cmd.Description)
				rows = append(rows, acNameStyle.Width(rowWidth).MaxWidth(rowWidth).Render(row))
			}
		}

		innerContent += "\n\n" + strings.Join(rows, "\n")
	}

	innerContent += "\n\n" + m.textInput.View()

	fileBar := m.renderFileBar()
	if fileBar != "" {
		innerContent += "\n\n" + strings.TrimSpace(fileBar)
	}

	inputBox := userBoxStyle.Width(m.width).MaxWidth(m.width).Render(innerContent)

	return fmt.Sprintf(
		"%s\n%s",
		m.viewport.View(),
		inputBox,
	)
}

func waitForIncomingMessage(conn *websocket.Conn) tea.Cmd {
	return func() tea.Msg {
		_, bytes, err := conn.ReadMessage()
		if err != nil {
			return err
		}

		var msg WSMessage
		if err := json.Unmarshal(bytes, &msg); err != nil {
			return WSMessage{Type: "error", Message: "Invalid JSON"}
		}
		return msg
	}
}

func main() {
	port := flag.String("port", "8000", "WebSocket server port")
	flag.Parse()

	u := url.URL{
		Scheme: "ws",
		Host:   "localhost:" + *port,
		Path:   "/ws/chat",
	}

	fmt.Println("Connecting to", u.String(), "...")

	conn, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		log.Fatal("Could not connect. Is the backend running?\nError:", err)
	}
	defer conn.Close()

	p := tea.NewProgram(initialModel(conn), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		log.Fatal("Error running program:", err)
	}
}
