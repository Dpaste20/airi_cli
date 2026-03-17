package commands

import (
	"fmt"
	"strings"
)

// toolEntry holds a tool name and a short description shown in /tools-list.
type toolEntry struct {
	Name        string
	Description string
}

// agentTools mirrors the TOOLS slice in server.py so the TUI can display them
// without making a network round-trip.
var agentTools = []struct {
	Category string
	Tools    []toolEntry
}{
	{
		Category: "System",
		Tools: []toolEntry{
			{"get_system_info", "Hardware / OS overview"},
			{"get_battery_status", "Battery level & charging state"},
			{"get_running_processes", "List active processes"},
			{"get_uptime", "System uptime"},
			{"get_disk_space", "Disk usage by mount"},
			{"get_system_logs", "Tail system journal logs"},
			{"get_active_connections", "Open network connections"},
			{"get_ip_info", "Public & local IP details"},
			{"run_system_diagnostic", "Full diagnostic report"},
			{"kill_processes", "Kill process by name / PID"},
			{"bash", "Run an arbitrary shell command"},
			{"shutdown_system", "Power off the machine"},
			{"restart_system", "Reboot the machine"},
			{"sleep_mode_system", "Suspend / sleep"},
		},
	},
	{
		Category: "📁  Files",
		Tools: []toolEntry{
			{"file_search", "Search files by name or pattern"},
			{"file_write", "Create or overwrite a file"},
			{"file_modify", "Edit an existing file"},
		},
	},
	{
		Category: "Web & URLs",
		Tools: []toolEntry{
			{"fetch_urls", "Fetch raw content from URLs"},
			{"open_url", "Open a URL in the default browser"},
			{"agent_browser", "Headless browser automation"},
		},
	},
	{
		Category: "News",
		Tools: []toolEntry{
			{"get_top_news", "Global top headlines"},
			{"get_topic_news", "News filtered by topic"},
			{"get_region_news", "News filtered by region"},
		},
	},
	{
		Category: " Gmail",
		Tools: []toolEntry{
			{"get_unread_emails", "Fetch unread inbox messages"},
			{"search_emails", "Search Gmail by query"},
			{"send_email", "Compose & send an email"},
			{"send_email_reply", "Reply to an existing thread"},
			{"create_draft_email", "Save a draft email"},
		},
	},
	{
		Category: "Google Drive",
		Tools: []toolEntry{
			{"list_drive_files", "List files in Drive"},
			{"search_drive_files", "Search Drive by name / MIME"},
			{"upload_to_drive", "Upload a local file to Drive"},
			{"download_from_drive", "Download a Drive file locally"},
		},
	},
	{
		Category: "Calendar & Tasks",
		Tools: []toolEntry{
			{"get_upcoming_events", "List upcoming calendar events"},
			{"create_event", "Create a calendar event"},
			{"delete_event", "Delete a calendar event"},
			{"list_tasks", "List Google Tasks"},
			{"add_task", "Add a new task"},
			{"complete_task", "Mark a task complete"},
			{"delete_task", "Delete a task"},
		},
	},
	{
		Category: "Maps",
		Tools: []toolEntry{
			{"map_search", "Search for a place on the map"},
			{"get_directions", "Get directions between two points"},
		},
	},
	{
		Category: "Telegram",
		Tools: []toolEntry{
			{"list_telegram_contacts", "List Telegram contacts"},
			{"send_telegram_message", "Send a Telegram message"},
		},
	},
	{
		Category: "Camera",
		Tools: []toolEntry{
			{"take_picture", "Capture a still photo"},
			{"take_timelapse", "Record a timelapse"},
			{"start_recording", "Start video recording"},
			{"stop_recording", "Stop video recording"},
			{"get_recording_status", "Check recording status"},
			{"list_captures", "List saved captures"},
			{"delete_capture", "Delete a capture by name"},
		},
	},
	{
		Category: "Music",
		Tools: []toolEntry{
			{"play_song", "Play a song by name"},
			{"play_playlist", "Play a playlist"},
			{"play_random", "Shuffle & play randomly"},
			{"pause_music", "Pause playback"},
			{"stop_music", "Stop playback"},
			{"next_song", "Skip to next track"},
			{"previous_song", "Go back one track"},
			{"set_volume", "Set playback volume (0-100)"},
			{"list_songs", "List available songs"},
		},
	},
	{
		Category: "Games",
		Tools: []toolEntry{
			{"launch_game", "Launch a game by name"},
			{"get_game_list", "List installed games"},
		},
	},
	{
		Category: "Cron",
		Tools: []toolEntry{
			{"get_cron_jobs", "List scheduled cron jobs"},
			{"add_cron_job", "Add a new cron job"},
			{"delete_cron_job", "Remove a cron job"},
		},
	},
	{
		Category: "Apps",
		Tools: []toolEntry{
			{"open_application", "Launch a desktop application"},
			{"get_current_datetime", "Get current date & time"},
		},
	},
	{
		Category: "Knowledge",
		Tools: []toolEntry{
			{"rag_search_tool", "Semantic search over local knowledge base"},
		},
	},
}

func ToolsList(args string) (Result, bool) {
	args = strings.TrimSpace(args)

	var sb strings.Builder

	totalTools := 0
	for _, group := range agentTools {
		totalTools += len(group.Tools)
	}

	sb.WriteString(fmt.Sprintf("**🛠️  Available Tools (%d)**\n\n", totalTools))

	filter := strings.ToLower(args)

	for _, group := range agentTools {
		var matched []toolEntry
		for _, t := range group.Tools {
			if filter == "" ||
				strings.Contains(strings.ToLower(t.Name), filter) ||
				strings.Contains(strings.ToLower(t.Description), filter) ||
				strings.Contains(strings.ToLower(group.Category), filter) {
				matched = append(matched, t)
			}
		}
		if len(matched) == 0 {
			continue
		}

		sb.WriteString(fmt.Sprintf("**%s**\n", group.Category))
		sb.WriteString("| Tool | Description |\n")
		sb.WriteString("|------|-------------|\n")
		for _, t := range matched {
			sb.WriteString(fmt.Sprintf("| `%s` | %s |\n", t.Name, t.Description))
		}
		sb.WriteString("\n")
	}

	return Result{
		ViewportMessage: sb.String(),
	}, true
}
