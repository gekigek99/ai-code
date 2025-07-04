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
SOURCE_DIRS = ["."]
EXCLUDE_DIRS = [".git", "ai-tools", "generation", "logs", "node_modules", "uploads", ".help.md", "package-lock.json", "tech"]
TREE_DIRS = ["."]
SYSTEM = r"""
You are a senior expo developer assistant. Analyze the following project files and help the user with described issues by rewriting, modifying, deleting, refactoring the code.

You are responsible for preparing the production website for public launch. Treat this as a mission-critical deployment: the website must be fully functional across both frontend and backend, rigorously secure, easily scalable, and robustly engineered for ongoing public usage. As you work, think methodically and comprehensively through every stage of implementation and do not overlook any implicit or supporting technical steps crucial to production readiness, even if not explicitly described in initial objectives.

For all code and configuration, provide deeply informative comments that clarify your logic, document all decisions, and support future maintenance or handovers. As you develop solutions, meticulously break down each objective into the smallest possible technical actions and address them one at a time, only proceeding once each is thorough and sound.

Implement all work following industry best practices for security, scalability, reliability, and long-term support. Proactively include essential techniques such as input validation, error management, logging, and operational monitoring. Consider compliance and data protection throughout. Design everything so that it is AI-ready: this means explicitly describing points in the codebase and architecture where AI models, logic, automation, analytics, or APIs can later be integrated with minimal refactoring, and providing hooks or scaffolding for such future capabilities.

For each decision and implementation step, provide full explanations and rationale, using clear code comments for technical documentation. Ensure that the delivered solution is exhaustive, production-grade, maintainable, and anticipates all reasonable future needs—especially. Do not provide summaries, layout, or graphic elements; concentrate on precise, actionable development and documentation only.

Make improvements or refactors. If a file should be deleted, write:"
+++ path/to/file.ext [DELETE] +++\n(no content needed)

If you want to create or overwrite a file, return the full updated code in this format:
+++ path/to/file.ext +++\n```\n<new content>\n```

Only output updated, new, or deleted files."""

PROMPT = r"""

"""

"""
THE NEXT PROMPT COMMAND IS FOR THE PRODUCTION WEBSITE, WE ARE GOIN PUBLIC SO EVERYTHING MUST BE WORKING ALSO ON THE BACKEND, BE SECURE SCALABLE AND DEFINITIVE.
It's essential you give a comprehensive explanation using code comments for future reference documentation.

implement user page config
"""

"""
THE NEXT PROMPT COMMAND IS FOR THE PRODUCTION WEBSITE, WE ARE GOIN PUBLIC SO EVERYTHING MUST BE WORKING ALSO ON THE BACKEND, BE SECURE SCALABLE AND DEFINITIVE.
It's essential you give a comprehensive explanation using code comments for future reference documentation.

implement payment page feature, it must be working also on the backend with database traking of payments and email notifications. the api key should be set in environment.
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

def export_md_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{data}")
    print(f"Saved selected files to: {filename}")


def write_or_warn_from_claude_output(output_text):
    export_md_file(
        output_text,
        filename=os.path.join(script_dir, f"{script_base_name}-clauderesponse.md")
    )

    # Match “+++ filename +++” or “+++ filename [TAG] +++”
    file_pattern = re.compile(
        r'^\+\+\+\s*'            # “+++ ” (leading)
        r'(.+?)'                 #   group(1)=filename
        r'(?:\s*\[([A-Z]+)\])?'  #   opt group(2)=TAG like UPDATE or DELETE
        r'\s*\+\+\+$',           # “ +++” (trailing)
        re.MULTILINE
    )
    parts = file_pattern.split(output_text)

    files_written = 0
    files_warned = 0

    for i in range(1, len(parts), 3):
        file_name, tag, content_block = parts[i], parts[i+1], parts[i+2]

        # If tag is DELETE, warn and skip writing
        if tag == "DELETE":
            print(f"DELETE: '{file_name}'")
            files_warned += 1
            continue

        # Extract code from ```…``` if present
        m = re.search(r'```(?:[^\n]*\n)?(.*?)```', content_block, re.DOTALL)
        content = m.group(1).rstrip() if m else content_block.strip()

        # Ensure directory exists (or use '.' for top-level files)
        target_dir = os.path.dirname(file_name) or '.'
        os.makedirs(target_dir, exist_ok=True)

        # Write file
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content + '\n')
        print(f"File written: {file_name}")
        files_written += 1

    print(f"\nSummary: {files_written} file(s) written, {files_warned} file(s) marked for deletion (not deleted).")


def get_directory_tree(base_dirs):
    print("Building directory tree...")
    project_root = os.path.abspath(os.getcwd())
    output_lines = []
    exclude_abs = [os.path.abspath(d) for d in EXCLUDE_DIRS]

    def is_nested(child, parents):
        child = os.path.abspath(child)
        for parent in parents:
            parent = os.path.abspath(parent)
            if os.path.commonpath([child, parent]) == parent and child != parent:
                return True
        return False

    def is_excluded(path):
        path = os.path.abspath(path)
        return any(os.path.commonpath([path, ex]) == ex for ex in exclude_abs)

    def walk_dir(path, prefix=""):
        if is_excluded(path):
            return  # Skip entire subtree
        try:
            entries = sorted(os.listdir(path))
        except Exception as e:
            output_lines.append(f"{prefix}[Error reading {path}]: {e}")
            return
        for i, entry in enumerate(entries):
            full_path = os.path.join(path, entry)
            if is_excluded(full_path):
                continue
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
        if is_excluded(dir):
            continue
        rel_name = os.path.relpath(dir, start=project_root)
        output_lines.append(f"\nDirectory tree structure for {rel_name}\\")
        walk_dir(dir)

    return "\n".join(output_lines)

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


def main():
    load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))
    run_claude = '-ai' in sys.argv
    run_readlast = '-readlast' in sys.argv
    force = '-f' in sys.argv

    # Check for uncommitted git changes
    if not force and has_uncommitted_changes():
        print("ERROR: You have uncommitted changes in your git repository.")
        print("Please commit or stash your changes before running this script.")
        sys.exit(1)

    if run_readlast:
        print("Applying files from last Claude response (ai-claude-clauderesponse.md)...")
        md_path = os.path.join(script_dir, f"{script_base_name}-clauderesponse.md")
        if not os.path.isfile(md_path):
            print(f"File not found: {md_path}")
            sys.exit(2)
        with open(md_path, encoding="utf-8") as f:
            data_response = f.read()
        if data_response.strip():
            write_or_warn_from_claude_output(data_response)
        else:
            print("Empty clauderesponse file.")
        return

    tree_dirs = get_directory_tree(TREE_DIRS)
    source_content = read_files(gather_source_files(SOURCE_DIRS))
    data_files = ""
    for path, content in source_content.items():
        data_files += f"\n--- {path} ---\n{content}\n"

    print(f"Input tokens [ESTIMATED]: {len(PROMPT) + len(tree_dirs) + len(data_files) // 4}")

    if not run_claude:
        export_md_file("\n".join([SYSTEM, PROMPT, tree_dirs, data_files]), os.path.join(script_dir, f"{script_base_name}-userfullprompt.md"))
        return

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("Sending request to Claude...")

    # Enable streaming to avoid the 10-minute non-streaming timeout
    response = client.messages.create(
        model=os.getenv("ANTHROPIC_CLAUDE_MODEL"),
        max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS")),
        temperature=1,
        system=SYSTEM,
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
        stream=True
    )

    # Stream and accumulate the response with proper handling for thinking and content
    data_response = ""
    thinking_content = ""
    raw_data = []  # Store all raw event data
    
    print("\nClaude's response:")
    
    for event in response:
        # Store raw event data
        raw_data.append({
            'type': event.type,
            'event': str(event)
        })
        
        try:
            if event.type == "content_block_delta":
                # Check if this is a thinking delta or regular content delta
                if hasattr(event, 'delta') and hasattr(event.delta, 'type'):
                    if event.delta.type == 'thinking_delta':
                        # Thinking content (internal reasoning)
                        thinking_chunk = event.delta.thinking
                        thinking_content += thinking_chunk
                        # Optionally display thinking process
                        # print(f"[THINKING: {thinking_chunk}]", end="", flush=True)
                    elif event.delta.type == 'text_delta':
                        # Regular content from Claude
                        chunk = event.delta.text
                        print(chunk, end="", flush=True)
                        data_response += chunk
                else:
                    # Fallback for unexpected delta structure
                    print(f"\n[DEBUG: Unexpected delta structure in {event.type}]")
            
            elif event.type == "message_start":
                # Message is starting
                print(f"\n[Message Start Event Data]: {event}")
            
            elif event.type == "message_stop":
                print(f"\n[Message Stop Event Data]: {event}")
                break
                
            elif event.type == "thinking_block_start":
                print("\n[Claude is thinking...]")
                
            elif event.type == "content_block_start":
                # Content block starting
                pass
                
            elif event.type == "content_block_stop":
                # Content block ending
                pass
            
            elif event.type == "message_delta":
                # Message delta - usually contains usage information
                pass
                
            else:
                # Log any other unexpected event types
                print(f"\n[DEBUG: Unhandled event type '{event.type}']")
                
        except AttributeError as e:
            # Handle unexpected event types gracefully
            print(f"\n[ERROR: AttributeError for event type '{event.type}': {e}]")
            print(f"[Event details: {event}]")
            continue

    # Save all data to separate files
    if thinking_content.strip():
        export_md_file(thinking_content, os.path.join(script_dir, f"{script_base_name}-thinking.md"))
        print(f"\n[Claude processed {len(thinking_content)} characters of internal reasoning]")
        
    if raw_data:
        raw_data_str = "\n\n".join([f"Event Type: {item['type']}\nData: {item['event']}" for item in raw_data])
        export_md_file(raw_data_str, os.path.join(script_dir, f"{script_base_name}-rawdata.md"))
        print(f"[Saved {len(raw_data)} raw events to file]")

    # Process the accumulated response
    if data_response.strip():
        print(f"\nProcessing Claude's response...")
        write_or_warn_from_claude_output(data_response)
    else:
        print("\nNo response received from Claude.")

if __name__ == "__main__":
    main()