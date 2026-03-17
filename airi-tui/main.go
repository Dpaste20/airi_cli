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
	"strings"
	"time"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/gorilla/websocket"

	"airi-tui/commands"
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
	senderStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("5")).Bold(true)
	aiStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("2")).Bold(true)
	errStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Bold(true)
	infoStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Italic(true)
	commandStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("214")).Bold(true)
	recordingStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Bold(true).Blink(true)

	userBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("5")).
			Padding(0, 1).
			Margin(0, 0)

	aiBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("2")).
			Padding(0, 1).
			Margin(0, 0)

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
			Foreground(lipgloss.Color("230")).
			Background(lipgloss.Color("236")).
			Padding(0, 1)

	statusConnectedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("42")).
				Bold(true)

	statusDisconnectedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("196")).
				Bold(true)

	statusLabelStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("240"))

	statusValueStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("255")).
				Bold(true)

	notificationStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("255")).
				Background(lipgloss.Color("63")).
				Bold(true).
				Padding(0, 2)

	acItemStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("252")).
			Background(lipgloss.Color("235")).
			Padding(0, 2)

	acSelectedStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("0")).
			Background(lipgloss.Color("63")).
			Bold(true).
			Padding(0, 2)

	acBorderStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("63"))

	fileBadgeStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("0")).
			Background(lipgloss.Color("214")).
			Bold(true).
			Padding(0, 1)

	fileBarStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("214")).
			Padding(0, 1)
)

var knownCommands = []string{
	"/help",
	"/attach",
	"/detach",
	"/save-session",
	"/resume-session",
	"/list-sessions",
	"/tools-list",
}

func getMatches(input string) []string {
	if !strings.HasPrefix(input, "/") || input == "" {
		return nil
	}
	lower := strings.ToLower(input)
	var matches []string
	for _, cmd := range knownCommands {
		if strings.HasPrefix(cmd, lower) && cmd != lower {
			matches = append(matches, cmd)
		}
	}
	return matches
}

// FileAttachment is the wire format sent to the backend.
type FileAttachment struct {
	Name     string `json:"name"`
	Content  string `json:"content"`   // base64-encoded bytes
	MimeType string `json:"mime_type"` // "text/plain" | "application/pdf"
}

// AttachedFile holds a file staged for the next outgoing message.
type AttachedFile struct {
	Name     string
	Path     string
	Content  string // base64
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

type model struct {
	conn               *websocket.Conn
	viewport           viewport.Model
	textInput          textinput.Model
	renderer           *glamour.TermRenderer
	messages           []string
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
	suggestions        []string
	suggestionIdx      int
	attachedFiles      []AttachedFile
}

// mimeFromExt returns the MIME type for supported file extensions.
func mimeFromExt(path string) (string, bool) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".txt", ".md", ".log", ".csv":
		return "text/plain", true
	case ".pdf":
		return "application/pdf", true
	default:
		return "", false
	}
}

// expandHome replaces a leading ~ with the user's home directory.
func expandHome(path string) string {
	if strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}

// loadAttachedFile reads the file at path, validates it, and returns an AttachedFile.
func loadAttachedFile(rawPath string) (AttachedFile, error) {
	path := expandHome(strings.TrimSpace(rawPath))

	mime, ok := mimeFromExt(path)
	if !ok {
		return AttachedFile{}, fmt.Errorf("unsupported file type — only .txt and .pdf are accepted")
	}

	info, err := os.Stat(path)
	if err != nil {
		return AttachedFile{}, fmt.Errorf("file not found: %s", path)
	}
	const maxBytes = 20 * 1024 * 1024 // 20 MB guard
	if info.Size() > maxBytes {
		return AttachedFile{}, fmt.Errorf("file too large (max 20 MB): %s", filepath.Base(path))
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

func initialModel(conn *websocket.Conn) model {
	ti := textinput.New()
	ti.Placeholder = "Ask Airi something... (type /help for commands)"
	ti.Focus()
	ti.CharLimit = 1000
	ti.Width = 20

	defaultWidth := 80
	vp := viewport.New(defaultWidth, 20)

	coloredLogo := aiStyle.Render(airiLogo)
	logoDisplay := lipgloss.PlaceHorizontal(defaultWidth, lipgloss.Center, coloredLogo)

	welcomeMsg := infoStyle.Render("\nConnected to Airi Terminal.\n")

	vp.SetContent(logoDisplay + "\n" + welcomeMsg)

	renderer, _ := glamour.NewTermRenderer(
		glamour.WithStandardStyle("dark"),
		glamour.WithWordWrap(74),
	)

	s := spinner.New()
	s.Spinner = spinner.MiniDot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("cyan"))

	return model{
		conn:               conn,
		textInput:          ti,
		viewport:           vp,
		renderer:           renderer,
		messages:           []string{logoDisplay, welcomeMsg},
		spinner:            s,
		isLoading:          false,
		sessionID:          "terminal_user",
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
	}
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
		return statusBarStyle.Width(m.width).Render(lipgloss.PlaceHorizontal(m.width, lipgloss.Center, msg))
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

	msgCount := fmt.Sprintf("%s %s",
		statusLabelStyle.Render("Messages:"),
		statusValueStyle.Render(fmt.Sprintf("%d", m.messageCount)))

	sessionInfo := fmt.Sprintf("%s %s",
		statusLabelStyle.Render("Session:"),
		statusValueStyle.Render(m.sessionID))

	genTokens := fmt.Sprintf("%s %s",
		statusLabelStyle.Render("Tokens:"),
		statusValueStyle.Render(fmt.Sprintf("%d", m.generatedTokens)))

	genTime := fmt.Sprintf("%s %s",
		statusLabelStyle.Render("Time:"),
		statusValueStyle.Render(fmt.Sprintf("%.2fs", m.generationTime.Seconds())))

	genSpeed := fmt.Sprintf("%s %s",
		statusLabelStyle.Render("Speed:"),
		statusValueStyle.Render(fmt.Sprintf("%.2f t/s", m.tokensPerSecond)))

	statusContent := lipgloss.JoinHorizontal(lipgloss.Left,
		connStatus,
		"  ",
		recStatus,
		"  ",
		msgCount,
		"  ",
		sessionInfo,
		"  ",
		genTokens,
		"  ",
		genTime,
		"  ",
		genSpeed,
	)

	return statusBarStyle.Width(m.width).Render(statusContent)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
	)

	switch msg := msg.(type) {

	case tea.KeyMsg:

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
				return m, tea.Tick(time.Millisecond*800, func(_ time.Time) tea.Msg {
					return clearNotificationMsg{}
				})
			}
			return m, nil

		case tea.KeyTab:
			if len(m.suggestions) > 0 {

				idx := m.suggestionIdx
				if idx < 0 {
					idx = 0
				}
				m.textInput.SetValue(m.suggestions[idx])
				m.textInput.CursorEnd()
				m.suggestions = nil
				m.suggestionIdx = -1
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
						m.messages = append(m.messages, errStyle.Render("Error reading audio: "+err.Error()))
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
					m.messages = append(m.messages, errStyle.Render("Error starting recording: "+err.Error()))
					m.isRecording = false
					return m, nil
				}
				return m, m.spinner.Tick
			}

		case tea.KeyEnter:

			if len(m.suggestions) > 0 && m.suggestionIdx >= 0 {
				m.textInput.SetValue(m.suggestions[m.suggestionIdx])
				m.textInput.CursorEnd()
				m.suggestions = nil
				m.suggestionIdx = -1
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

			// ── /attach <path> ───────────────────────────────────────────
			if strings.HasPrefix(input, "/attach") {
				m.textInput.SetValue("")
				rawPath := strings.TrimSpace(strings.TrimPrefix(input, "/attach"))

				cmdLabel := commandStyle.Render("You:")
				cmdDisplay := userBoxStyle.Width(m.width - 6).Render(
					fmt.Sprintf("%s %s", cmdLabel, input),
				)
				m.messages = append(m.messages, cmdDisplay)
				m.messageCount++

				if rawPath == "" {
					msg := "Usage: `/attach <path>`  —  supported: `.txt`, `.pdf`"
					m.messages = append(m.messages, aiBoxStyle.Width(m.width-6).Render(
						aiStyle.Render("Airi:")+"\n"+m.renderMarkdown(msg),
					))
					m.messageCount++
					m.viewport.SetContent(strings.Join(m.messages, "\n"))
					m.viewport.GotoBottom()
					return m, nil
				}

				af, err := loadAttachedFile(rawPath)
				if err != nil {
					errBox := aiBoxStyle.BorderForeground(lipgloss.Color("9")).Width(m.width - 6).Render(
						aiStyle.Render("Airi:") + "\n" + errStyle.Render("❌ "+err.Error()),
					)
					m.messages = append(m.messages, errBox)
					m.messageCount++
					m.viewport.SetContent(strings.Join(m.messages, "\n"))
					m.viewport.GotoBottom()
					return m, nil
				}

				// Prevent duplicate attachments
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
				m.messages = append(m.messages, aiBoxStyle.Width(m.width-6).Render(
					aiStyle.Render("Airi:")+"\n"+m.renderMarkdown(okMsg),
				))
				m.messageCount++
				m.viewport.SetContent(strings.Join(m.messages, "\n"))
				m.viewport.GotoBottom()
				return m, nil
			}

			// ── /detach ──────────────────────────────────────────────────
			if input == "/detach" {
				m.textInput.SetValue("")
				cmdDisplay := userBoxStyle.Width(m.width - 6).Render(
					fmt.Sprintf("%s %s", commandStyle.Render("You:"), input),
				)
				m.messages = append(m.messages, cmdDisplay)
				m.messageCount++

				if len(m.attachedFiles) == 0 {
					m.messages = append(m.messages, aiBoxStyle.Width(m.width-6).Render(
						aiStyle.Render("Airi:")+"\n"+infoStyle.Render("No files are currently attached."),
					))
				} else {
					m.attachedFiles = nil
					m.messages = append(m.messages, aiBoxStyle.Width(m.width-6).Render(
						aiStyle.Render("Airi:")+"\n"+m.renderMarkdown("🗑 All attachments cleared."),
					))
				}
				m.messageCount++
				m.viewport.SetContent(strings.Join(m.messages, "\n"))
				m.viewport.GotoBottom()
				return m, nil
			}

			if result, handled := commands.Dispatch(input, m.sessionID, m.messages); handled {
				m.textInput.SetValue("")

				cmdLabel := commandStyle.Render("You:")
				cmdDisplay := userBoxStyle.Width(m.width - 6).Render(
					fmt.Sprintf("%s %s", cmdLabel, input),
				)
				m.messages = append(m.messages, cmdDisplay)
				m.messageCount++

				if result.ViewportMessage != "" {
					rendered := m.renderMarkdown(result.ViewportMessage)
					style := aiBoxStyle
					if result.IsError {
						style = aiBoxStyle.BorderForeground(lipgloss.Color("9"))
					}
					responseBox := style.Width(m.width - 6).Render(
						aiStyle.Render("Airi:") + "\n" + rendered,
					)
					m.messages = append(m.messages, responseBox)
					m.messageCount++
				}

				if result.NewSessionID != "" {
					m.sessionID = result.NewSessionID
				}

				if result.RestoredMessages != nil {
					m.messages = result.RestoredMessages
					m.messageCount = len(result.RestoredMessages)
				}

				var notifCmd tea.Cmd
				if result.Notification != "" {
					m.notification = result.Notification
					notifCmd = tea.Tick(2*time.Second, func(_ time.Time) tea.Msg {
						return clearNotificationMsg{}
					})
				}

				m.viewport.SetContent(strings.Join(m.messages, "\n"))
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

			label := senderStyle.Render("You:")
			userBubbleText := input
			if len(m.attachedFiles) > 0 {
				var names []string
				for _, af := range m.attachedFiles {
					names = append(names, "📎 "+af.Name)
				}
				userBubbleText += "\n" + infoStyle.Render(strings.Join(names, "  "))
			}
			displayInput := userBoxStyle.Width(m.width - 6).Render(
				fmt.Sprintf("%s %s", label, userBubbleText),
			)

			m.messages = append(m.messages, displayInput)
			m.messageCount++

			m.currentAiChunk = ""
			m.isLoading = true

			m.generatedTokens = 0
			m.generationTime = 0
			m.tokensPerSecond = 0.0

			content := strings.Join(m.messages, "\n")
			header := aiStyle.Render("Airi:")
			content += "\n" + header + " " + m.spinner.View()

			m.viewport.SetContent(content)
			m.viewport.GotoBottom()

			// Build file attachments for the wire payload.
			var wireFiles []FileAttachment
			for _, af := range m.attachedFiles {
				wireFiles = append(wireFiles, FileAttachment{
					Name:     af.Name,
					Content:  af.Content,
					MimeType: af.MimeType,
				})
			}
			m.attachedFiles = nil // clear after staging

			req := ChatRequest{
				Message:         input,
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
			return m, tea.Batch(sendCmd, m.spinner.Tick)
		}

	case clearNotificationMsg:
		m.notification = ""

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - 5
		m.textInput.Width = msg.Width

		coloredLogo := aiStyle.Render(airiLogo)
		centeredLogo := lipgloss.PlaceHorizontal(msg.Width, lipgloss.Center, coloredLogo)

		if len(m.messages) > 0 {
			m.messages[0] = centeredLogo
		}

		m.renderer, _ = glamour.NewTermRenderer(
			glamour.WithStandardStyle("dark"),
			glamour.WithWordWrap(msg.Width-8),
		)

		content := strings.Join(m.messages, "\n")
		if m.currentAiChunk != "" {
			renderedChunk := m.renderMarkdown(m.currentAiChunk)

			fullContent := aiStyle.Render("Airi:") + "\n" + renderedChunk
			boxedContent := aiBoxStyle.Width(m.width - 6).Render(fullContent)
			content += "\n" + boxedContent
		} else if m.isLoading {
			header := aiStyle.Render("Airi:")
			content += "\n" + header + " " + m.spinner.View()
		}
		m.viewport.SetContent(content)

	case spinner.TickMsg:
		if m.isLoading {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)

			if m.currentAiChunk == "" {
				content := strings.Join(m.messages, "\n")
				header := aiStyle.Render("Airi:")
				content += "\n" + header + " " + m.spinner.View()
				m.viewport.SetContent(content)
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

				formattedVoice := fmt.Sprintf("%s %s", senderStyle.Render("You:"), voiceText)
				displayInput := userBoxStyle.Width(m.width - 6).Render(formattedVoice)

				m.messages = append(m.messages, displayInput)
				m.messageCount++
				m.pendingVoiceInput = voiceText
				m.awaitingVoiceChunk = false

				content := strings.Join(m.messages, "\n")
				header := aiStyle.Render("Airi:")
				content += "\n" + header + " " + m.spinner.View()
				m.viewport.SetContent(content)
				m.viewport.GotoBottom()
			} else {

				m.currentAiChunk += msg.Content

				m.generatedTokens = len(m.currentAiChunk) / 4
				elapsed := time.Since(m.generationStart).Seconds()
				if elapsed > 0 {
					m.tokensPerSecond = float64(m.generatedTokens) / elapsed
				}

				renderedContent := m.renderMarkdown(m.currentAiChunk)

				fullContent := aiStyle.Render("Airi:") + "\n" + renderedContent
				boxedContent := aiBoxStyle.Width(m.width - 6).Render(fullContent)

				displayMessages := append([]string{}, m.messages...)
				displayMessages = append(displayMessages, boxedContent)

				m.viewport.SetContent(strings.Join(displayMessages, "\n"))
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

			renderedContent := m.renderMarkdown(m.currentAiChunk)

			fullContent := aiStyle.Render("Airi:") + "\n" + renderedContent
			boxedContent := aiBoxStyle.Width(m.width - 6).Render(fullContent)

			m.messages = append(m.messages, boxedContent)
			m.messageCount++
			m.currentAiChunk = ""
			m.isLoading = false
			m.pendingVoiceInput = ""
			m.isSpeaking = true

		case "speech_stopped":
			m.isSpeaking = false

		case "error":
			errMsg := errStyle.Render("Error: " + msg.Message)
			m.messages = append(m.messages, errMsg)
			m.viewport.SetContent(strings.Join(m.messages, "\n"))
			m.viewport.GotoBottom()
			m.isLoading = false
			m.isSpeaking = false

		default:
			if msg.Error != "" {
				m.messages = append(m.messages, errStyle.Render("System Error: "+msg.Error))
				m.viewport.SetContent(strings.Join(m.messages, "\n"))
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
		}
		badges = append(badges, fileBadgeStyle.Render(icon+" "+af.Name))
	}
	hint := fileBarStyle.Render("ctrl+d to clear")
	return strings.Join(badges, " ") + "  " + hint + "\n"
}

func (m model) View() string {
	if m.err != nil {
		return errStyle.Render(fmt.Sprintf("\nFatal Error: %v\nRestart the application.", m.err))
	}

	if m.showHelp {
		title := helpTitleStyle.Render("Commands / Short Cuts :")

		content := fmt.Sprintf(`
%s

%s
  /attach <path>          Attach a .txt or .pdf file to next message
  /detach                 Remove all staged attachments
  /save-session [name]    Save current conversation
  /resume-session [name]  Restore a saved conversation
  /resume-session         List all saved sessions
  /tools-list             Show all available agent tools
  /tools-list [keyword]   Filter tools by name or category
  /help                   Show this screen

%s
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

	dropdownBlock := ""
	if len(m.suggestions) > 0 {
		var rows []string
		for i, cmd := range m.suggestions {
			if i == m.suggestionIdx {
				rows = append(rows, acSelectedStyle.Render(cmd))
			} else {
				rows = append(rows, acItemStyle.Render(cmd))
			}
		}
		inner := strings.Join(rows, "\n")
		dropdownBlock = acBorderStyle.Render(inner) + "\n"
	}

	return fmt.Sprintf(
		"%s\n%s\n%s%s%s",
		m.viewport.View(),
		m.renderStatusBar(),
		dropdownBlock,
		m.renderFileBar(),
		m.textInput.View(),
	) + "\n"
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
