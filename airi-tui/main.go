package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"strings"

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

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"strings"

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
	senderStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("5")).Bold(true)
	aiStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("2")).Bold(true)
	errStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Bold(true)
	infoStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Italic(true)
)

type ChatRequest struct {
	Message         string `json:"message"`
	SessionID       string `json:"session_id"`
	SearchKnowledge bool   `json:"search_knowledge"`
}

type WSMessage struct {
	Type    string `json:"type"`
	Content string `json:"content"`
	Message string `json:"message"`
	Error   string `json:"error"`
}

type model struct {
	conn           *websocket.Conn
	viewport       viewport.Model
	textInput      textinput.Model
	renderer       *glamour.TermRenderer
	messages       []string
	currentAiChunk string
	err            error
	spinner        spinner.Model
	isLoading      bool
}

func initialModel(conn *websocket.Conn) model {
	ti := textinput.New()
	ti.Placeholder = "Ask Airi something..."
	ti.Focus()
	ti.CharLimit = 156
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
		conn:      conn,
		textInput: ti,
		viewport:  vp,
		renderer:  renderer,
		messages:  []string{logoDisplay, welcomeMsg},
		spinner:   s,
		isLoading: false,
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

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
	)

	switch msg := msg.(type) {

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			return m, tea.Quit
		case tea.KeyEnter:
			input := m.textInput.Value()
			if input == "" {
				return m, nil
			}

			m.messages = append(m.messages, senderStyle.Render("You: ")+input)

			m.currentAiChunk = ""
			m.isLoading = true

			content := strings.Join(m.messages, "\n")
			header := aiStyle.Render("Airi:")
			content += "\n" + header + " " + m.spinner.View()

			m.viewport.SetContent(content)
			m.viewport.GotoBottom()

			req := ChatRequest{
				Message:         input,
				SessionID:       "terminal_user",
				SearchKnowledge: false,
			}
			sendCmd := func() tea.Msg {
				err := m.conn.WriteJSON(req)
				if err != nil {
					return err
				}
				return nil
			}

			m.textInput.SetValue("")
			return m, tea.Batch(sendCmd, m.spinner.Tick)
		}

	case tea.WindowSizeMsg:
		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - 4
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
		}

	case WSMessage:
		switch msg.Type {
		case "start":
			m.isLoading = true

		case "chunk":
			m.currentAiChunk += msg.Content
			renderedContent := m.renderMarkdown(m.currentAiChunk)
			header := aiStyle.Render("Airi:") + "\n"

			displayMessages := append([]string{}, m.messages...)
			displayMessages = append(displayMessages, header+renderedContent)

			m.viewport.SetContent(strings.Join(displayMessages, "\n"))
			m.viewport.GotoBottom()

		case "end":
			renderedContent := m.renderMarkdown(m.currentAiChunk)
			header := aiStyle.Render("Airi:") + "\n"

			m.messages = append(m.messages, header+renderedContent)
			m.currentAiChunk = ""
			m.isLoading = false

		case "error":
			errMsg := errStyle.Render("Error: " + msg.Message)
			m.messages = append(m.messages, errMsg)
			m.viewport.SetContent(strings.Join(m.messages, "\n"))
			m.viewport.GotoBottom()
			m.isLoading = false

		default:
			if msg.Error != "" {
				m.messages = append(m.messages, errStyle.Render("System Error: "+msg.Error))
				m.viewport.SetContent(strings.Join(m.messages, "\n"))
				m.viewport.GotoBottom()
				m.isLoading = false
			}
		}

		return m, waitForIncomingMessage(m.conn)

	case error:
		m.err = msg
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
		"%s\n\n%s",
		m.viewport.View(),
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
