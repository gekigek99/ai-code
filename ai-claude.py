import os
import datetime
import sys
import re
from anthropic import Anthropic
from dotenv import load_dotenv
import subprocess
import yaml
import fnmatch

# python ./ai-code/ai-claude.py -ai

# Load configuration from YAML
def load_config():
    with open(os.path.join(os.path.dirname(__file__), "ai-claude-prompt.yaml")) as f:
        return yaml.safe_load(f)

config = load_config()
SOURCE_DIRS = config.get("source_dirs", ["."])
TREE_DIRS = config.get("tree_dirs") or SOURCE_DIRS
EXCLUDE_PATTERNS = config.get("exclude_patterns", [])
SYSTEM = config.get("system", "")
PROMPT = config.get("prompt", "")

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
script_dir_name = os.path.basename(script_dir)
script_file_name = os.path.basename(script_path)
script_base_name = os.path.splitext(script_file_name)[0]  # name without extension

# Create output directory for all generated files
output_dir = os.path.join(script_dir, script_base_name)
os.makedirs(output_dir, exist_ok=True)


def normalize_path(path):
    """Normalize path for consistent pattern matching."""
    # Convert to absolute path and normalize separators
    abs_path = os.path.abspath(path)
    # Convert to forward slashes for consistent matching
    return abs_path.replace('\\', '/')


def normalize_pattern(pattern):
    """Normalize pattern for consistent matching."""
    # Convert backslashes to forward slashes and remove trailing slashes
    pattern = pattern.replace('\\', '/')
    # Keep track of whether it was a directory pattern
    is_dir = pattern.endswith('/')
    pattern = pattern.rstrip('/')
    return pattern, is_dir


def path_matches_pattern(path, pattern):
    """Check if a path matches a specific pattern, supporting multi-level paths."""
    # Normalize both for comparison
    norm_path = path.replace('\\', '/')
    norm_pattern, is_dir_pattern = normalize_pattern(pattern)
    
    # For multi-level patterns (containing /), check if the pattern appears in the path
    if '/' in norm_pattern:
        # Check if the pattern matches the end of the path or appears as a complete segment
        path_parts = norm_path.split('/')
        pattern_parts = norm_pattern.split('/')
        
        # Check all possible positions where the pattern could match
        for i in range(len(path_parts) - len(pattern_parts) + 1):
            # Check if pattern matches at this position
            match = True
            for j, pattern_part in enumerate(pattern_parts):
                if not fnmatch.fnmatch(path_parts[i + j], pattern_part):
                    match = False
                    break
            if match:
                return True
        return False
    else:
        # Single-level pattern - check against each path component
        path_parts = norm_path.split('/')
        for part in path_parts:
            if fnmatch.fnmatch(part, norm_pattern):
                return True
        return False


def is_excluded(path, base_dir=None):
    """Check if a path matches any exclude pattern."""
    # Normalize the path for matching
    norm_path = normalize_path(path)
    
    # Get the base name of the path (last component)
    base_name = os.path.basename(path)
    
    # If base_dir is provided, get relative path for matching
    if base_dir:
        base_norm = normalize_path(base_dir)
        try:
            # Get relative path from base directory
            rel_path = os.path.relpath(path, base_dir).replace('\\', '/')
        except ValueError:
            # If paths are on different drives on Windows, use full path
            rel_path = norm_path
    else:
        # Try to get relative path from current directory
        try:
            rel_path = os.path.relpath(path).replace('\\', '/')
        except ValueError:
            rel_path = norm_path
    
    # Check each exclude pattern
    for pattern in EXCLUDE_PATTERNS:
        # Normalize the pattern
        norm_pattern, is_dir_pattern = normalize_pattern(pattern)
        
        # Check for multi-level path patterns (containing /)
        if '/' in norm_pattern:
            # Check against relative path
            if path_matches_pattern(rel_path, pattern):
                return True
            # Also check against the normalized full path
            if path_matches_pattern(norm_path, pattern):
                return True
        else:
            # Single-level pattern
            
            # For .env* pattern, special handling
            if norm_pattern.startswith('.') and '*' in norm_pattern:
                if fnmatch.fnmatch(base_name, norm_pattern):
                    return True
            
            # Check base name
            if fnmatch.fnmatch(base_name, norm_pattern):
                return True
            
            # For directories, check if any parent matches
            if os.path.isdir(path) or is_dir_pattern:
                path_parts = rel_path.split('/')
                for part in path_parts:
                    if fnmatch.fnmatch(part, norm_pattern):
                        return True
            
            # For files in excluded directories
            if not os.path.isdir(path):
                path_parts = rel_path.split('/')
                for i in range(len(path_parts) - 1):  # Exclude the filename
                    if fnmatch.fnmatch(path_parts[i], norm_pattern):
                        return True
    
    return False


def gather_source_files(dirs):
    print("\nGathering  source directories...")
    all_files = set()
    
    for d in dirs:
        # Normalize directory path
        base_dir = os.path.abspath(d)
        print(f"Scanning: {base_dir}")
        
        # Walk through directory tree
        for root, dirnames, filenames in os.walk(base_dir):
            # Filter out excluded directories before descending
            # Create a copy of dirnames to iterate over
            dirs_to_check = dirnames[:]
            dirnames.clear()  # Clear the original list
            
            for dirname in dirs_to_check:
                dirpath = os.path.join(root, dirname)
                if not is_excluded(dirpath, base_dir):
                    dirnames.append(dirname)  # Add back non-excluded directories
                else:
                    # print(f"  Excluding directory: {os.path.relpath(dirpath, base_dir)}")
                    continue

            # Check files
            for filename in filenames:
                filepath = os.path.join(root, filename)
                
                # Skip if excluded
                if is_excluded(filepath, base_dir):
                    # print(f"  Excluding file: {os.path.relpath(filepath, base_dir)}")
                    continue
                    
                all_files.add(filepath)
    
    return sorted(list(all_files))


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

        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                contents[path] = f.read()
        except Exception as e:
            print(f"Error reading {path}: {e}")
            contents[path] = f"[Error reading file: {e}]"
    return contents


def export_md_file(data, filename):
    """Export markdown file to the output directory."""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{data}")
    print(f"Saved to: {rel_path(filepath)}")

# Log file for prompts
def log_prompt(content):
    """Append the given prompt content to the log file with timestamp."""
    log_path = os.path.join(output_dir, 'prompt.log')
    timestamp = datetime.datetime.now().isoformat()
    try:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"\n=== Prompt logged at {timestamp} ===\n" + content + "\n")
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")

def write_or_warn_from_claude_output(output_text):
    """Process Claude's response and write files."""
    # Save the raw response to output directory
    export_md_file(output_text, "clauderesponse.md")

    # Match "+++ filename +++" or "+++ filename [TAG] +++"
    file_pattern = re.compile(
        r'^\+\+\+\s*'            # "+++ " (leading)
        r'(.+?)'                 #   group(1)=filename
        r'(?:\s*\[([A-Z]+)\])?'  #   opt group(2)=TAG like UPDATE or DELETE
        r'\s*\+\+\+$',           # " +++" (trailing)
        re.MULTILINE
    )
    parts = file_pattern.split(output_text)

    files_written = 0
    files_deleted = 0

    for i in range(1, len(parts), 3):
        file_name, tag, content_block = parts[i], parts[i+1], parts[i+2]

        if tag == "DELETE":
            # Delete the file if it exists
            if os.path.exists(file_name):
                try:
                    os.remove(file_name)
                    files_deleted += 1
                    print(f"File deleted: {file_name}")
                except Exception as e:
                    print(f"Error deleting {file_name}: {e}")
            else:
                print(f"File not found (for deletion): {file_name}")
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

    print(f"\nSummary: {files_written} file(s) written, {files_deleted} file(s) deleted.")


def get_directory_tree(base_dirs):
    print("Building directory tree...")
    project_root = os.path.abspath(os.getcwd())
    output_lines = []

    def is_nested(child, parents):
        child = os.path.abspath(child)
        for parent in parents:
            parent = os.path.abspath(parent)
            try:
                if os.path.commonpath([child, parent]) == parent and child != parent:
                    return True
            except ValueError:
                # Paths on different drives on Windows
                continue
        return False

    def walk_dir(path, prefix="", base_dir=None):
        if is_excluded(path, base_dir or path):
            return  # Skip entire subtree
        try:
            entries = sorted(os.listdir(path))
        except Exception as e:
            output_lines.append(f"{prefix}[Error reading {path}]: {e}")
            return
        
        # Filter out excluded entries
        filtered_entries = []
        for entry in entries:
            full_path = os.path.join(path, entry)
            if not is_excluded(full_path, base_dir or path):
                filtered_entries.append(entry)
        
        for i, entry in enumerate(filtered_entries):
            full_path = os.path.join(path, entry)
            is_last = (i == len(filtered_entries) - 1)
            connector = "└── " if is_last else "├── "
            output_lines.append(f"{prefix}{connector}{entry}")
            if os.path.isdir(full_path):
                extension = "    " if is_last else "│   "
                walk_dir(full_path, prefix + extension, base_dir or path)

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
        output_lines.append(f"\nDirectory tree structure for {rel_name}/")
        walk_dir(dir, base_dir=dir)

    dir_tree = "\n".join(output_lines)
    print(dir_tree)
    return dir_tree


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


def rel_path(abs_path) -> str:
    """Convert absolute path to relative path from current directory"""
    rel_path = os.path.relpath(abs_path).replace('\\', '/')
    # Ensure it starts with ./
    if not rel_path.startswith('./'):
        rel_path = './' + rel_path
    return rel_path

def main():
    load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))
    run_claude = '-ai' in sys.argv
    run_readlast = '-readlast' in sys.argv
    force = '-f' in sys.argv

    if run_readlast:
        print(f"Applying files from last Claude response ({output_dir}/clauderesponse.md)...")
        md_path = os.path.join(output_dir, "clauderesponse.md")
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
        data_files += f"\n--- {rel_path(path)} ---\n{content}\n"

    print(f"Input tokens [ESTIMATED]: {len(PROMPT) + len(tree_dirs) + len(data_files) // 4}")
    
    if not run_claude:
        # Save the full prompt to output directory
        export_md_file("\n".join([SYSTEM, PROMPT, tree_dirs, data_files]), "userfullprompt.md")
        return
    
    # Check for uncommitted git changes
    if not force and has_uncommitted_changes():
        print("ERROR: You have uncommitted changes in your git repository.")
        print("Please commit or stash your changes before running this script.")
        sys.exit(1)

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

    # Save all data to output directory
    if thinking_content.strip():
        export_md_file(thinking_content, "thinking.md")
        print(f"\n[Claude processed {len(thinking_content)} characters of internal reasoning]")
        
    if raw_data:
        raw_data_str = "\n\n".join([f"Event Type: {item['type']}\nData: {item['event']}" for item in raw_data])
        export_md_file(raw_data_str, "rawdata.md")
        print(f"[Saved {len(raw_data)} raw events to file]")

    # Process the accumulated response
    if data_response.strip():
        print(f"\nProcessing Claude's response...")
        write_or_warn_from_claude_output(data_response)
    else:
        print("\nNo response received from Claude.")
    
    # log prompt for history tracking
    log_prompt(PROMPT)


if __name__ == "__main__":
    main()