import os
import datetime
import sys
import re
import base64
import mimetypes
from anthropic import Anthropic
import subprocess
import yaml
import fnmatch
import fitz

# python ./ai-code/ai-claude.py -ai

# Load configuration from YAML
def load_config():
    with open(os.path.join(os.path.dirname(__file__), "ai-claude-prompt.yaml"), "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()
ANTHROPIC = config.get("ANTHROPIC", {})
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


def is_pdf_file(path):
    """Check if a file is a PDF based on extension and optionally file header."""
    # Check extension first for performance
    if not path.lower().endswith('.pdf'):
        return False
    
    # Optionally verify PDF header for extra safety
    try:
        with open(path, 'rb') as f:
            header = f.read(5)
            return header == b'%PDF-'
    except Exception:
        # If we can't read the file, assume it's not a valid PDF
        return False


def extract_text_from_pdf(path):
    """
    Extract text content from a PDF file using PyMuPDF.
    
    Returns a string containing all extracted text, or an error message if extraction fails.

    We are extracting PDF text, for full PDF support use the anthropic PDF support (https://docs.anthropic.com/en/docs/build-with-claude/pdf-support). Enable citations for pdf image analysis.
    """    
    try:
        # Open the PDF document
        pdf_document = fitz.open(path)
        
        # Extract text from all pages
        text_content = []
        
        # Add PDF metadata as header comment
        text_content.append(f"[PDF Document: {os.path.basename(path)}]")
        text_content.append(f"[Pages: {pdf_document.page_count}]")
        
        # Extract metadata if available
        metadata = pdf_document.metadata
        if metadata:
            if metadata.get('title'):
                text_content.append(f"[Title: {metadata['title']}]")
            if metadata.get('author'):
                text_content.append(f"[Author: {metadata['author']}]")
        
        text_content.append("[Content:]")
        text_content.append("")
        
        # Extract text from each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Add page separator for multi-page documents
            if pdf_document.page_count > 1:
                text_content.append(f"\n└─ Page {page_num + 1}")
            
            # Extract text from the page
            page_text = page.get_text()
            
            # Clean up the text: remove excessive whitespace but preserve structure
            page_text = re.sub(r'\n{3,}', '\n\n', page_text)  # Limit consecutive newlines
            page_text = page_text.strip()
            
            if page_text:
                text_content.append(page_text)
            else:
                text_content.append("[No text content on this page]")
        
        # Close the document
        pdf_document.close()
        
        # Join all content
        full_text = '\n'.join(text_content)
        
        # If no meaningful text was extracted, indicate this
        if not any(line.strip() and not line.startswith('[') for line in text_content):
            return "[PDF contains no extractable text content - may be scanned/image-based]"
        
        return full_text
        
    except Exception as e:
        # Return detailed error information for debugging
        return f"[Error extracting PDF text: {type(e).__name__}: {str(e)}]"


def is_binary_file(path, include_pdfs=False):
    """
    Check if a file is binary (non-text).
    
    Args:
        path: File path to check
        include_pdfs: If True and PDF support is enabled, PDFs are not considered binary
    
    Returns:
        True if file is binary, False if it's text or (when enabled) a PDF
    """
    # If PDF support is requested and this is a PDF, it's not binary for our purposes
    if include_pdfs and is_pdf_file(path):
        return False
    
    try:
        with open(path, 'rb', encoding='utf-8') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return True
        return False
    except Exception:
        return True  # treat as binary if unreadable


def read_files(file_paths, read_pdfs=False):
    """
    Read contents of files, with optional PDF text extraction.
    
    Args:
        file_paths: List of file paths to read
        read_pdfs: If True, extract text from PDF files
    
    Returns:
        Dictionary mapping file paths to their contents
    """
    print("Reading files...")
    
    contents = {}
    pdf_count = 0
    
    for path in file_paths:
        if not os.path.isfile(path):
            print(f"Skipped: {path} (not found)")
            continue

        # Check if this is a PDF and we should read it
        if read_pdfs and is_pdf_file(path):
            pdf_count += 1
            contents[path] = extract_text_from_pdf(path)
            continue

        # Check if file is binary
        if is_binary_file(path, include_pdfs=read_pdfs):
            contents[path] = "[binary content]"
            continue

        # Read as regular text file
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                contents[path] = f.read()
        except Exception as e:
            print(f"Error reading {path}: {e}")
            contents[path] = f"[Error reading file: {e}]"
    
    if read_pdfs and pdf_count > 0:
        print(f"Extracted text from {pdf_count} PDF file(s)")
    
    return contents


def parse_image_arguments():
    """
    Parse command line arguments to extract image paths from -img flags.
    
    Returns:
        List of image paths specified via -img flags
    """
    image_paths = []
    
    # Look for -img flags followed by paths
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '-img' and i + 1 < len(sys.argv):
            # Next argument should be the image path
            img_path = sys.argv[i + 1]
            # Convert to absolute path for consistency
            abs_img_path = os.path.abspath(img_path)
            image_paths.append(abs_img_path)
            i += 2  # Skip both -img and the path
        else:
            i += 1
    
    return image_paths


def determine_image_media_type(file_path):
    """
    Determine the MIME type of an image file based on its extension and content.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        String containing the MIME type (e.g., 'image/jpeg', 'image/png')
        
    Raises:
        ValueError: If the file is not a supported image type
    """
    # First check the file extension
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Validate it's an image type and is supported by Claude
    supported_types = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'
    }
    
    if mime_type and mime_type.lower() in supported_types:
        return mime_type.lower()
    
    # If mimetypes couldn't determine it, try based on extension
    ext = os.path.splitext(file_path)[1].lower()
    extension_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg', 
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    
    if ext in extension_map:
        return extension_map[ext]
    
    # Try to read file header to determine type
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
            
        # Check common image headers
        if header.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        elif header.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
            return 'image/gif'
        elif header.startswith(b'RIFF') and b'WEBP' in header:
            return 'image/webp'
    except Exception:
        pass
    
    raise ValueError(f"Unsupported or unrecognized image type for file: {file_path}")


def process_images(image_paths):
    """
    Process image files by reading, validating, and encoding them for Claude API.
    
    Args:
        image_paths: List of file paths to process
        
    Returns:
        List of dictionaries containing processed image data with format:
        {
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': 'image/jpeg',
                'data': 'base64_encoded_data'
            },
            'path': '/path/to/original/file'  # Added for debugging/logging
        }
    """
    processed_images = []
    
    if not image_paths:
        return processed_images
        
    print(f"\nProcessing {len(image_paths)} image(s)...")
    
    for img_path in image_paths:
        try:
            # Validate that the file exists
            if not os.path.isfile(img_path):
                print(f"ERROR: Image file not found: {img_path}")
                continue
                
            # Determine the media type
            try:
                media_type = determine_image_media_type(img_path)
            except ValueError as e:
                print(f"ERROR: {e}")
                continue
                
            # Read and encode the image file
            try:
                with open(img_path, 'rb') as f:
                    image_data = f.read()
                    
                # Validate file size (Claude has limits, typically around 5MB per image)
                file_size_mb = len(image_data) / (1024 * 1024)
                if file_size_mb > 5:
                    print(f"WARNING: Image {img_path} is {file_size_mb:.1f}MB, which may exceed Claude's size limits")
                    
                # Encode to base64
                base64_data = base64.b64encode(image_data).decode('utf-8')
                
                # Create the structure expected by Claude API
                image_content = {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': media_type,
                        'data': base64_data
                    },
                    'path': img_path  # Keep for logging/debugging
                }
                
                processed_images.append(image_content)
                print(f"✓ Processed: {rel_path(img_path)} ({media_type}, {file_size_mb:.1f}MB)")
                
            except Exception as e:
                print(f"ERROR: Failed to read/encode image {img_path}: {e}")
                continue
                
        except Exception as e:
            print(f"ERROR: Unexpected error processing image {img_path}: {e}")
            continue
    
    if processed_images:
        print(f"Successfully processed {len(processed_images)} image(s)")
    else:
        print("No images were successfully processed")
        
    return processed_images


def estimate_image_tokens(processed_images):
    """
    Estimate the token count for images.
    
    Based on Claude documentation, images typically consume ~1600 tokens each,
    but this can vary significantly based on image size, complexity, and format.
    This provides a rough estimate for planning purposes.
    An other way to calculate them: tokens = (witdh px * height px)/750
    
    Args:
        processed_images: List of processed image dictionaries
        
    Returns:
        Integer estimate of tokens consumed by images
    """
    if not processed_images:
        return 0
        
    # Base estimate per image - Claude documentation suggests ~1600 tokens
    # but this can vary from 85 to 1600+ depending on image characteristics
    base_tokens_per_image = 1200  # Conservative middle estimate
    
    total_tokens = 0
    
    for img in processed_images:
        # Get the base64 data length as a rough proxy for image complexity
        try:
            data_length = len(img['source']['data'])
            
            # Adjust estimate based on data size
            # Larger base64 data usually means more complex/higher resolution images
            if data_length > 500000:  # Large image
                tokens = 1600
            elif data_length > 200000:  # Medium image  
                tokens = 1200
            else:  # Small image
                tokens = 800
                
            total_tokens += tokens
            
        except KeyError:
            # Fallback to base estimate if we can't access the data
            total_tokens += base_tokens_per_image
    
    return total_tokens


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

    # Match "\+\+\+ filename \+\+\+" or "\+\+\+ filename [TAG] \+\+\+"
    file_pattern = re.compile(
        r'^\+\+\+\s*'            # "\+\+\+ " (leading)
        r'(.+?)'                 #   group(1)=filename
        r'(?:\s*\[([A-Z]+)\])?'  #   opt group(2)=TAG like UPDATE or DELETE
        r'\s*\+\+\+$',           # " \+\+\+" (trailing)
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

        # Extract code
        m = re.search(r'\`\`\`(?:[^\n]*\n)?(.*?)\`\`\`', content_block, re.DOTALL)
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
    # Parse command line flags
    run_claude = '-ai' in sys.argv
    run_readlast = '-readlast' in sys.argv
    force = '-f' in sys.argv
    read_pdfs = '-pdf' in sys.argv  # New flag for PDF support
    
    # Parse image paths from command line arguments
    image_paths = parse_image_arguments()
    
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

    # Build directory tree and read files with PDF support if enabled
    tree_dirs = get_directory_tree(TREE_DIRS)
    source_content = read_files(gather_source_files(SOURCE_DIRS), read_pdfs=read_pdfs)
    
    # Process images if any were specified
    processed_images = process_images(image_paths) if image_paths else []
    
    # Format file contents for prompt
    data_files = ""
    for path, content in source_content.items():
        data_files += f"\n--- {rel_path(path)} ---\n{content}\n"

    # Calculate token estimates
    text_tokens = len(PROMPT) + len(tree_dirs) + len(data_files) // 4
    image_tokens = estimate_image_tokens(processed_images)
    total_tokens = text_tokens + image_tokens
    
    print(f"Input tokens [ESTIMATED]: {total_tokens}")
    if image_tokens > 0:
        print(f"  - Text tokens: {text_tokens}")
        print(f"  - Image tokens: {image_tokens} ({len(processed_images)} image(s))")
    
    if not run_claude:
        # Save the full prompt to output directory
        export_md_file("\n".join([SYSTEM, PROMPT, tree_dirs, data_files]), "userfullprompt.md")
        return
    
    # Check for uncommitted git changes
    if not force and has_uncommitted_changes():
        print("ERROR: You have uncommitted changes in your git repository.")
        print("Please commit or stash your changes before running this script.")
        sys.exit(1)

    client = Anthropic(api_key=ANTHROPIC["API_KEY"])
    print("Sending request to Claude...")

    # Build the message content array with images and text
    message_content = []
    
    # Add images first (recommended order for Claude API)
    for img in processed_images:
        message_content.append({
            'type': 'image',
            'source': img['source']  # Contains type, media_type, and data
        })
    
    # Add text content
    message_content.extend([
        {"type": "text", "text": PROMPT},
        {"type": "text", "text": f"Directory tree structure: {tree_dirs}"},
        {"type": "text", "text": data_files}
    ])

    # Enable streaming to avoid the 10-minute non-streaming timeout
    response = client.messages.create(
        model=ANTHROPIC["CLAUDE_MODEL"],
        max_tokens=int(ANTHROPIC["MAX_TOKENS"]),
        temperature=1,
        system=SYSTEM,
        messages=[
            {"role": "user", "content": message_content}
        ],
        tools=[],
        thinking={
            "type": "enabled",
            "budget_tokens": int(ANTHROPIC["MAX_TOKENS_THINK"]),
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
