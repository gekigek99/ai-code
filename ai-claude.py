import os
import sys
import re
import glob
from anthropic import Anthropic
from dotenv import load_dotenv
import subprocess

# TODO: Claude ai send full code folder structure with prompt
# TODO: repository git a parte ed import come submodule

# Define the directories to scan
SOURCE_DIRS = ["src\\",
]
EXCLUDE_DIRS = [
]
PROMPT = """
if user data is not inserted in config screen, prompt for user data insertion at first startup + add a start logo of 2 seconds at beginning
"""

def gather_source_files(dirs):
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
    print("\n".join(all_files))
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

def export_md_file(source_content, filename=f"{os.path.splitext(os.path.basename(__file__))[0]}-monofile.md"):
    with open(filename, 'w', encoding='utf-8') as f:
        for path, content in source_content.items():
            f.write(f"## SOURCE: {os.path.basename(path)}\n")
            f.write("```\n" + content + "\n```\n\n")
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

def main():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    run_claude = '-ai' in sys.argv

    # Check for uncommitted git changes
    if has_uncommitted_changes():
        print("ERROR: You have uncommitted changes in your git repository.")
        print("Please commit or stash your changes before running this script.")
        sys.exit(1)

    print("Scanning source directories...")
    source_files = gather_source_files(SOURCE_DIRS)
    source_content = read_files(source_files)

    print(f"Input tokens [ESTIMATED]: {len(PROMPT) + len("\n".join([s+c for s, c in source_content.items()])) // 4}")

    if not run_claude:
        export_md_file(source_content)
        return

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("Sending request to Claude...")

    data_files = ""
    for path, content in source_content.items():
        data_files += f"\n--- {path} ---\n{content}\n"

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
        if os.path.basename(os.path.abspath(__file__)) in line:
            continue
        changes_count += 1
    
    if changes_count != 0:
        return True
    return False


if __name__ == "__main__":
    main()
