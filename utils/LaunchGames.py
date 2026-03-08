import os
import subprocess

from agno.tools import tool

TUI_GAMES_DIR = os.path.join(os.getcwd(), "TUIGames")


@tool
def get_game_list(max_results: int = 100, **kwargs) -> list[str]:
    """Returns a list of available Terminal games

    Args:
        max_results (int): The maximum number of games to return (default: 100) .
    """
    if not os.path.exists(TUI_GAMES_DIR):
        return ["Error: TUIGames directory not found."]

    games = []
    try:
        for item in os.listdir(TUI_GAMES_DIR):
            item_path = os.path.join(TUI_GAMES_DIR, item)

            if os.path.isdir(item_path):
                binary_path = os.path.join(item_path, item)
                if os.path.isfile(binary_path) and os.access(binary_path, os.X_OK):
                    games.append(item)
    except Exception as e:
        return [f"Error reading games directory: {e}"]

    result = games[:max_results] if games else ["No games found."]
    return result


@tool
def launch_game(game_name: str) -> str:
    """Launches the specified TUI game in a new terminal window."""
    binary_path = os.path.join(TUI_GAMES_DIR, game_name, game_name)

    if not os.path.exists(binary_path):
        return f"Error: Game binary not found at {binary_path}"

    if not os.access(binary_path, os.X_OK):
        return f"Error: Game binary at {binary_path} is not executable. Run 'chmod +x {binary_path}'"

    try:
        subprocess.Popen(
            ["gnome-terminal", "--", binary_path],
            cwd=os.path.join(TUI_GAMES_DIR, game_name),
        )
        return f"Successfully launched {game_name}."
    except FileNotFoundError:
        try:
            subprocess.Popen(
                ["xterm", "-e", binary_path], cwd=os.path.join(TUI_GAMES_DIR, game_name)
            )
            return f"Successfully launched {game_name} using xterm."
        except Exception as e:
            return f"Failed to launch terminal emulator: {e}"
    except Exception as e:
        return f"Failed to launch game: {e}"
