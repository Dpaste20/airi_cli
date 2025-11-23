import shutil


def ascii_art(file_path):
    try:
        terminal_width = shutil.get_terminal_size().columns
    except OSError:
        terminal_width = 80

    with open(file_path, "r") as file:
        content = file.read()

    lines = content.splitlines()

    max_width = 0
    if lines:
        max_width = max(len(line) for line in lines)

    padding = (terminal_width - max_width) // 2

    padding_str = " " * max(0, padding)

    print("\n")
    for line in lines:
        print(f"{padding_str}{line}")
    print("\n")


ascii_art("ascii_art.txt")
