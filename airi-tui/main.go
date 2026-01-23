package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/gorilla/websocket"
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

	thinkingEnabledStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("214")).
				Bold(true)

	thinkingDisabledStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("39")).
				Bold(true)
)

type ChatRequest struct {
	Message         string `json:"message,omitempty"`
	SessionID       string `json:"session_id"`
	SearchKnowledge bool   `json:"search_knowledge"`
	AudioData       string `json:"audio_data,omitempty"`
	Action          string `json:"action,omitempty"`
}

type WSMessage struct {
	Type           string  `json:"type"`
	Content        string  `json:"content"`
	Message        string  `json:"message"`
	Error          string  `json:"error"`
	TokenCount     int     `json:"token_count"`
	GenerationTime float64 `json:"generation_time"`
}

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
	thinkingMode       bool
	generatedTokens    int
	generationStart    time.Time
	generationTime     time.Duration
	tokensPerSecond    float64
	isRecording        bool
	recordCmd          *exec.Cmd
	awaitingVoiceChunk bool
	pendingVoiceInput  string
	isSpeaking         bool
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
		glamour.WithWordWrap(80),
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
		thinkingMode:       true,
		generatedTokens:    0,
		generationTime:     0,
		tokensPerSecond:    0.0,
		isRecording:        false,
		awaitingVoiceChunk: false,
		pendingVoiceInput:  "",
		isSpeaking:         false,
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		textinput.Blink,
		waitForIncomingMessage(m.conn),
	)
}

func (m model) renderMarkdown(text string) string {
	tr, err := m.renderer.Render(text)
	if err != nil {
		return text
	}
	return tr
}

func (m model) renderStatusBar() string {
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

	var thinkMode string
	if m.thinkingMode {
		thinkMode = fmt.Sprintf("%s %s",
			statusLabelStyle.Render("Mode:"),
			thinkingEnabledStyle.Render("🧠 Think"))
	} else {
		thinkMode = fmt.Sprintf("%s %s",
			statusLabelStyle.Render("Mode:"),
			thinkingDisabledStyle.Render("⚡ Fast"))
	}

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
		thinkMode,
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

func (m *model) detectThinkingModeChange(content string) {
	if strings.Contains(content, "Thinking mode disabled") || strings.Contains(content, "Responses will be faster") {
		m.thinkingMode = false
	} else if strings.Contains(content, "Thinking mode enabled") || strings.Contains(content, "Model will reason") {
		m.thinkingMode = true
	}
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
	)

	switch msg := msg.(type) {

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
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
			input := m.textInput.Value()
			if input == "" {
				return m, nil
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

			var displayInput string
			if strings.HasPrefix(input, "/") {
				displayInput = commandStyle.Render("You: ") + input
			} else {
				displayInput = senderStyle.Render("You: ") + input
			}

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

			req := ChatRequest{
				Message:         input,
				SessionID:       m.sessionID,
				SearchKnowledge: false,
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
			glamour.WithWordWrap(msg.Width-2),
		)

		content := strings.Join(m.messages, "\n")
		if m.currentAiChunk != "" {
			renderedChunk := m.renderMarkdown(m.currentAiChunk)
			header := aiStyle.Render("Airi:") + "\n"
			content += "\n" + header + renderedChunk
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

				displayInput := senderStyle.Render("You: ") + voiceText
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

				m.detectThinkingModeChange(msg.Content)

				renderedContent := m.renderMarkdown(m.currentAiChunk)
				header := aiStyle.Render("Airi:") + "\n"

				displayMessages := append([]string{}, m.messages...)
				displayMessages = append(displayMessages, header+renderedContent)

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

			m.detectThinkingModeChange(m.currentAiChunk)

			renderedContent := m.renderMarkdown(m.currentAiChunk)
			header := aiStyle.Render("Airi:") + "\n"

			m.messages = append(m.messages, header+renderedContent)
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

	return m, tea.Batch(tiCmd, vpCmd)
}

func (m model) View() string {
	if m.err != nil {
		return errStyle.Render(fmt.Sprintf("\nFatal Error: %v\nRestart the application.", m.err))
	}

	return fmt.Sprintf(
		"%s\n%s\n%s",
		m.viewport.View(),
		m.renderStatusBar(),
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
	u := url.URL{Scheme: "ws", Host: "localhost:8000", Path: "/ws/chat"}
	fmt.Println("Connecting to", u.String(), "...")

	conn, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		log.Fatal("Could not connect. Is the backend running?\nError: ", err)
	}
	defer conn.Close()

	p := tea.NewProgram(initialModel(conn), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		log.Fatal("Error running program:", err)
	}
}
