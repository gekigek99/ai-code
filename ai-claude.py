import os
import sys
import re
import glob
from anthropic import Anthropic
from dotenv import load_dotenv
import subprocess

# to run:
# setup SOURCE_DIRS, EXCLUDE_DIRS, TREE_DIRS, PROMPT
# python ai-claude.py -ai

# Define the directories to scan
SOURCE_DIRS = ["src", "tech"]
EXCLUDE_DIRS = []
TREE_DIRS = ["src", "tech"]
PROMPT = """
if user data is not inserted in config screen, prompt for user data insertion at first startup + add a start logo of 2 seconds at beginning
"""

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
script_dir_name = os.path.basename(script_dir)
script_file_name = os.path.basename(script_path)
script_base_name = os.path.splitext(script_file_name)[0]  # name without extension

def gather_source_files(dirs):
    print("\nScanning source directories...")
    all_files = set()
    for d in dirs:
        # Grab everything recursively
        d = d.replace("/", "\\")
        candidates = glob.glob(os.path.join(d, "**", "*"), recursive=True)
        for f in candidates:
            if not os.path.isfile(f):
                continue
            # Exclude files in EXCLUDE_DIRS
            if any(os.path.commonpath([f, ex]) == ex for ex in EXCLUDE_DIRS):
                continue
            all_files.add(f)
    return list(all_files)

def is_binary_file(path):
    try:
        with open(path, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return True
        return False
    except Exception:
        return True  # treat as binary if unreadable
    
def read_files(file_paths):
    print("Reading files...")
    contents = {}
    for path in file_paths:
        if not os.path.isfile(path):
            print(f"Skipped: {path} (not found)")
            continue

        if is_binary_file(path):
            contents[path] = "[binary content]"
            continue

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            contents[path] = f.read()
    return contents

def export_md_file(data, filename=os.path.join(script_dir, f"{script_base_name}-userfullprompt.md")):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{data}")
    print(f"Saved selected files to: {filename}")

def write_or_warn_from_claude_output(output_text):
    file_pattern = re.compile(r'^--- (.+?) ---$', re.MULTILINE)
    parts = file_pattern.split(output_text)
    
    files_written = 0
    files_warned = 0

    for i in range(1, len(parts), 2):
        file_name = parts[i].strip()
        content = parts[i+1].strip()

        if "[DELETE]" in file_name.upper():
            file_to_delete = file_name.replace("[DELETE]", "").strip()
            print(f"WARNING: Claude suggests deleting '{file_to_delete}'")
            files_warned += 1
            continue

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content + '\n')
        print(f"File written: {file_name}")
        files_written += 1

    print(f"\nSummary: {files_written} file(s) written, {files_warned} file(s) marked for deletion (not deleted).")

def get_directory_tree(base_dirs):
    print("Building directory tree...")
    project_root = os.path.abspath(os.getcwd())
    output_lines = []

    def is_nested(child, parents):
        child = os.path.abspath(child)
        for parent in parents:
            parent = os.path.abspath(parent)
            if os.path.commonpath([child, parent]) == parent and child != parent:
                return True
        return False

    def walk_dir(path, prefix=""):
        try:
            entries = sorted(os.listdir(path))
        except Exception as e:
            output_lines.append(f"{prefix}[Error reading {path}]: {e}")
            return
        for i, entry in enumerate(entries):
            full_path = os.path.join(path, entry)
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            output_lines.append(f"{prefix}{connector}{entry}")
            if os.path.isdir(full_path):
                extension = "    " if is_last else "│   "
                walk_dir(full_path, prefix + extension)

    abs_dirs = [os.path.abspath(d) for d in base_dirs]

    # Check for missing directories
    missing_dirs = [d for d in abs_dirs if not os.path.isdir(d)]
    if missing_dirs:
        output_lines.append("\nERROR: The following directory(ies) do not exist:")
        for d in missing_dirs:
            rel_path = os.path.relpath(d, start=project_root)
            output_lines.append(f" - {rel_path}")
        return "\n".join(output_lines)

    # Filter out nested dirs
    roots_to_print = []
    for d in abs_dirs:
        if not is_nested(d, roots_to_print):
            roots_to_print.append(d)

    for dir in roots_to_print:
        rel_name = os.path.relpath(dir, start=project_root)
        output_lines.append(f"\nDirectory tree structure for {rel_name}\\")
        walk_dir(dir)

    return "\n".join(output_lines)

def main():
    load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))
    run_claude = '-ai' in sys.argv

    # Check for uncommitted git changes
    if has_uncommitted_changes():
        print("ERROR: You have uncommitted changes in your git repository.")
        print("Please commit or stash your changes before running this script.")
        sys.exit(1)

    tree_dirs = get_directory_tree(TREE_DIRS)
    source_content = read_files(gather_source_files(SOURCE_DIRS))
    data_files = ""
    for path, content in source_content.items():
        data_files += f"\n--- {path} ---\n{content}\n"

    print(f"Input tokens [ESTIMATED]: {len(PROMPT) + len(tree_dirs) + len(data_files) // 4}")

    if not run_claude:
        export_md_file("\n".join([PROMPT, tree_dirs, data_files]))
        return

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("Sending request to Claude...")

    response = client.messages.create(
        model=os.getenv("ANTHROPIC_CLAUDE_MODEL"),
        max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS")),
        temperature=1,
        system=(
                "You are a senior expo developer assistant. Analyze the following project files and help the user with described issues by rewriting, modifying, deleting, refactoring the code\n"
                "Make improvements or refactors. If a file should be deleted, write:\n"
                "--- path/to/file.ext [DELETE] ---\n(no content needed).\n"
                "If you want to create or overwrite a file, return the full updated code in this format:\n"
                "--- path/to/file.ext ---\n<new content>\n"
                "Only output updated, new, or deleted files.\n\n"
            ),
        messages=[
            {"role": "user", "content": [{"type": "text", "text": PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": f"Directory tree structure: {tree_dirs}"}]},
            {"role": "user", "content": [{"type": "text", "text": data_files}]}
            ],
        tools=[],
        thinking={
            "type": "enabled",
            "budget_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS_THINK")),
        },
        # setup for stream=True?
    )

    print(f"Response received [I: {response.usage.input_tokens}; O: {response.usage.output_tokens}]. Processing...")
    data_response = ""
    for block in response.content:
        if block.type == "text":
            data_response += block.text
        elif block.type == "tool_use":
            print(f"{block.type}:\n{block}")
        elif block.type == "thinking":
            print(f"{block.type}:\n{block.thinking}")
        else:
            print(f"[{block.type} block ignored: {block}]")
    write_or_warn_from_claude_output(data_response)

def has_uncommitted_changes():
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        capture_output=True,
        text=True,
    )
    changes = result.stdout.strip().splitlines()
    
    changes_count = 0
    for line in changes:
        if not line.strip():
            continue

        if script_dir_name in line:
            continue

        changes_count += 1

    return changes_count != 0

if __name__ == "__main__":
    main()
