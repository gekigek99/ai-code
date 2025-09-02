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
from typing import List, Any
from dataclasses import dataclass, asdict
from typing import Any, List, Optional

# python ./ai-code/ai-claude.py -ai

# Load configuration from YAML
def load_config():
    with open(os.path.join(os.path.dirname(__file__), "ai-claude-prompt.yaml"), "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()
ANTHROPIC = config.get("ANTHROPIC", {})
SOURCE = config.get("source")
TREE_DIRS = config.get("tree_dirs") or SOURCE
EXCLUDE_PATTERNS = config.get("exclude_patterns")
SYSTEM = config.get("system") + """

# ----- file output patterns ----- #

If a file should be deleted, write:
+++++ ./path/to/file.ext [DELETE] +++++
  (no content needed)
+++++

If you want to create or overwrite a file, return the full updated code in this format:
+++++ ./path/to/file.ext +++++
  <new content>
+++++

Only output updated, new, or deleted files."""
PROMPT = config.get("prompt")


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
script_dir_name = os.path.basename(script_dir)
script_file_name = os.path.basename(script_path)
script_base_name = os.path.splitext(script_file_name)[0]  # name without extension

# Create output directory for all generated files
output_dir = os.path.join(script_dir, script_base_name)
os.makedirs(output_dir, exist_ok=True)


@dataclass
class FileData:
    file_type: str
    path_abs: str
    path_rel: str
    extension: str
    ai_share: bool

    data: Any
    data_type: str
    data_size: int
    media_type: str

    ai_interpretable: bool
    ai_data_converted: str
    ai_data_converted_type: str
    ai_data_tokens: int

    @classmethod
    def get(
        cls, 
        files: List['FileData'], 
        path: str
    ) -> Optional['FileData']:
        """
        Search for a FileData in the list by absolute or relative path.
        Returns the FileData if found, otherwise None.
        """
        for f in files:
            if f.path_abs == path or f.path_rel == path:
                return f
        return None

def is_excluded(path, base_dir=None):
    """Check if a path matches any exclude pattern."""

    def path_matches_pattern(path, pattern):
        """Check if a path matches a specific pattern, supporting multi-level paths."""
        # Normalize both for comparison
        norm_path = path.replace('\\', '/')
        norm_pattern = pattern.replace('\\', '/').rstrip('/')
        
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

    # Normalize the path for matching
    norm_path = os.path.abspath(path).replace('\\', '/')
    
    # Get the base name of the path (last component)
    base_name = os.path.basename(path).replace('\\', '/')
    
    # If base_dir is provided, get relative path for matching
    if base_dir:
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
        pattern = pattern.replace('\\', '/')
        is_dir_pattern = pattern.endswith('/')
        norm_pattern = pattern.rstrip('/') 
        
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


def add_source(files_to_ai: List[FileData], SOURCE, ai_shared_file_types=[]) -> List[FileData]:
    """adds source"""

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
    
    def is_file_pdf(path):
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

    def is_file_bin(path):
        """
        Check if a file is binary (non-text).
        
        Returns:
            True if file is binary, False if it's text
        """
        
        try:
            with open(path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:
                    return True
            return False
        except Exception as e:
            return True  # treat as binary if unreadable

    
    print("\nGathering source files...")
    abs_file_paths = set()
    
    for s in SOURCE:
        abs_entry = os.path.abspath(s)
        if not os.path.exists(abs_entry):
            print(f"WARNING: Source entry not found: {s}")
            continue

        # If it's a file, just add it
        if os.path.isfile(abs_entry):
            if not is_excluded(abs_entry):
                abs_file_paths.add(abs_entry)
            else:
                print(f"Excluding file: {s}")
            continue

        # If it's a directory, walk it
        if os.path.isdir(abs_entry):
            print(f"Scanning directory: {abs_entry}")
            for root, dirnames, filenames in os.walk(abs_entry):
                # Filter directories
                dirs_to_check = dirnames[:]
                dirnames.clear()
                for dirname in dirs_to_check:
                    dirpath = os.path.join(root, dirname)
                    if not is_excluded(dirpath, abs_entry):
                        dirnames.append(dirname)
                # Files
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    if not is_excluded(filepath, abs_entry):
                        abs_file_paths.add(filepath)
        else:
            print(f"Skipping unknown source: {s}")
    
    abs_file_paths = sorted(list(abs_file_paths))

    print("\nReading files...")  
    
    for abs_path in abs_file_paths:
        from pathlib import Path

        rel_path = f"./{os.path.relpath(abs_path, os.getcwd())}"

        if not os.path.isfile(abs_path):
            print(f"Skipped: {rel_path} (not found)")
            continue

        _, extension = os.path.splitext(abs_path)
            
        # Check if file is binary
        if is_file_bin(abs_path):
            # BINARY FILE
            with open(abs_path, "rb") as f:
                file_data = f.read()

            if is_file_pdf(abs_path):
                # PDF file
                file_data_text = extract_text_from_pdf(abs_path)
                files_to_ai.append(
                    FileData(
                        file_type="pdf",
                        path_abs=abs_path,
                        path_rel=rel_path,
                        extension=extension,
                        ai_share="pdf" in ai_shared_file_types,

                        data=file_data,
                        data_type="byte",
                        data_size=len(file_data),
                        media_type="pdf",

                        ai_interpretable=True,
                        ai_data_converted=file_data_text,
                        ai_data_converted_type="str",
                        ai_data_tokens=len(file_data_text)//4
                    )
                )
            
            else:
                # Binary file
                files_to_ai.append(
                    FileData(
                        file_type="bin",
                        path_abs=abs_path,
                        path_rel=rel_path,
                        extension=extension,
                        ai_share=True,

                        data=file_data,
                        data_type="byte",
                        data_size=len(file_data),
                        media_type="pdf",

                        ai_interpretable=False,
                        ai_data_converted="[BINARY CONTENT]",
                        ai_data_converted_type="str",
                        ai_data_tokens=20
                    )
                )

        else:
            # TEXT FILE
            with open(abs_path, 'r', encoding='utf-8') as f:
                file_data_text = f.read()
            
            files_to_ai.append(
                FileData(
                    file_type="text",
                    path_abs=abs_path,
                    path_rel=rel_path,
                    extension=extension,
                    ai_share=True,

                    data=file_data_text,
                    data_type="str",
                    data_size=len(file_data_text),
                    media_type="text",

                    ai_interpretable=True,
                    ai_data_converted=file_data_text,
                    ai_data_converted_type="str",
                    ai_data_tokens=len(file_data_text)//4
                )
            )
    
    return files_to_ai


def add_images(files_to_ai: List[FileData]) -> List[FileData]:
    """
    Parse command line arguments to extract image data from -img flags.
    """

    def get_image_media_type(file_path):
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

    def get_image_data(img_path):
        """
        Get base64 encoded string of image

        Returns binary data, base64 data, image size (bytes)
        """

        # Validate that the file exists
        if not os.path.isfile(img_path):
            print(f"ERROR: Image file not found: {img_path}")
            return bytes(), "", -1
            

        # Read and encode the image file
        try:
            with open(img_path, 'rb') as f:
                image_data = f.read()
                
            # Validate file size (Claude has limits, typically around 5MB per image)
            file_size_mb = len(image_data) / (1024 * 1024)
            if file_size_mb > 5:
                print(f"WARNING: Image {img_path} is {file_size_mb:.1f}MB, which may exceed Claude's size limits")
                
            img_data_base64 = base64.b64encode(image_data).decode('utf-8')
            
            return image_data, img_data_base64, len(image_data)
            
        except Exception as e:
            print(f"ERROR: Failed to read/encode image {img_path}: {e}")
            return bytes(), "", -1

    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '-img' and i + 1 < len(sys.argv):
            img_path = sys.argv[i + 1]
            abs_path = os.path.abspath(img_path)
            _, extension = os.path.splitext(abs_path)
            img_data, img_data_base64, size = get_image_data(img_path)
            files_to_ai.append(
                FileData(
                    file_type="image",
                    path_abs=os.path.abspath(img_path),
                    path_rel=img_path,
                    extension=extension,
                    ai_share=True, # default true if it was included as flag
                    
                    data=img_data,
                    data_type="binary",
                    data_size=size,
                    media_type=get_image_media_type(img_path),

                    ai_interpretable=True,
                    ai_data_converted=img_data_base64,
                    ai_data_converted_type="base64",
                    ai_data_tokens=1600 if len(img_data_base64) > 500000 else 1200 if len(img_data_base64) > 200000 else 800
                )
            )
            i += 2
        else:
            i += 1
    
    return files_to_ai


def export_md_file(data, filename):
    """Export markdown file to the output directory."""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{data}")
    print(f"Saved to: {os.path.relpath(filepath).replace('\\', '/')}")

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
    """Process Claude's response and write or delete files."""
    # Save the raw response to output directory
    export_md_file(output_text, "clauderesponse.md")

    # Regex: match "+++++ filename [TAG] +++++" ... content ... "+++++"
    block_pattern = re.compile(
        r'^\+\+\+\+\+\s*'          # opening +++++
        r'(.+?)'                   # group(1) = filename
        r'(?:\s*\[([A-Z]+)\])?'    # optional group(2) = TAG (DELETE, UPDATE, etc.)
        r'\s*\+\+\+\+\+\s*\n'      # end of header line
        r'(.*?)'                   # group(3) = content (non-greedy)
        r'^\+\+\+\+\+$',           # closing +++++
        re.MULTILINE | re.DOTALL
    )

    matches = block_pattern.findall(output_text)

    files_written = 0
    files_deleted = 0

    for file_name, tag, content_block in matches:
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

        content = content_block.strip()

        # Ensure directory exists (or use '.' for top-level files)
        target_dir = os.path.dirname(file_name) or '.'
        os.makedirs(target_dir, exist_ok=True)

        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(content + '\n')
            print(f"File written: {file_name}")
            files_written += 1
        except Exception as e:
            print(f"Error writing {file_name}: {e}")

    print(f"\nSummary: {files_written} file(s) written, {files_deleted} file(s) deleted.")


def get_directory_tree(base_dirs, files_to_ai: List[FileData] = None):
    """
    Build a directory tree for the given base_dirs. If `files_to_ai` contains
    a file path that is being printed, that line will be wrapped in ANSI green
    and will include a token estimate when available.
    """

    print("Building directory tree...")
    project_root = os.path.abspath(os.getcwd())
    output_lines = []

    # Normalize files_to_ai into a dict keyed by normalized absolute path
    files_to_ai = files_to_ai or []
    files_to_ai_norm = {
        os.path.normcase(os.path.abspath(f.path_abs)): f
        for f in files_to_ai
    }

    def is_nested(child, parents):
        child = os.path.abspath(child)
        for parent in parents:
            parent = os.path.abspath(parent)
            try:
                if os.path.commonpath([child, parent]) == parent and child != parent:
                    return True
            except ValueError:
                # different drives on Windows
                continue
        return False

    def walk_dir(path, prefix="", base_dir=None):
        # Skip excluded directories early
        if is_excluded(path, base_dir or path):
            return
        try:
            entries = sorted(os.listdir(path))
        except Exception as e:
            output_lines.append(f"{prefix}[Error reading {path}]: {e}")
            return

        # Filter entries by exclusion rules
        filtered_entries = []
        for entry in entries:
            abs_path = os.path.join(path, entry)
            if not is_excluded(abs_path, base_dir or path):
                filtered_entries.append(entry)

        for i, entry in enumerate(filtered_entries):
            abs_path = os.path.join(path, entry)
            is_last = (i == len(filtered_entries) - 1)
            connector = "└── " if is_last else "├── "

            # Build entry text
            base_name = os.path.basename(abs_path)
            entry_text = base_name

            # Try to get file data from files_to_ai_norm using consistent normalization
            filedata = files_to_ai_norm.get(os.path.normcase(os.path.abspath(abs_path)))

            if os.path.isfile(abs_path):
                try:
                    file_size = os.path.getsize(abs_path)
                    size_kb = file_size / 1024
                except Exception:
                    size_kb = 0.0

                if filedata:
                    # Show token estimate if available
                    token_est = getattr(filedata, "ai_data_tokens", None)
                    if token_est is None:
                        token_str = "~N/A tokens"
                    else:
                        token_str = f"~{token_est:5.0f} tokens"
                    # Color files that are in files_to_ai
                    entry_text = f"\033[32m{base_name} {base_name.ljust(20)[len(base_name):]} [{size_kb:5.1f} KB | {token_str}]\033[0m"
                else:
                    # File not indexed in files_to_ai -> show size only and mark as not indexed
                    entry_text = f"{base_name} {base_name.ljust(20)[len(base_name):]} [{size_kb:5.1f} KB ]"

            else:
                # Directories and other non-files: just display name
                entry_text = entry

            output_lines.append(f"{prefix}{connector}{entry_text}")

            # Recurse into directories
            if os.path.isdir(abs_path):
                extension = "    " if is_last else "│   "
                walk_dir(abs_path, prefix + extension, base_dir or path)

    # Normalize base_dirs and check existence
    abs_dirs = [os.path.abspath(d) for d in base_dirs]

    missing_dirs = [d for d in abs_dirs if not os.path.isdir(d)]
    if missing_dirs:
        output_lines.append("\nERROR: The following directory(ies) do not exist:")
        for d in missing_dirs:
            rel_path = os.path.relpath(d, start=project_root)
            output_lines.append(f" - {rel_path}")
        dir_tree = "\n".join(output_lines)
        print(dir_tree)
        return dir_tree

    # Remove nested entries so we don't print duplicates
    roots_to_print = []
    for d in abs_dirs:
        if not is_nested(d, roots_to_print):
            roots_to_print.append(d)

    for root_dir in roots_to_print:
        if is_excluded(root_dir):
            continue
        rel_name = os.path.relpath(root_dir, start=project_root)
        output_lines.append(f"\nDirectory tree structure for {rel_name}/")
        walk_dir(root_dir, base_dir=root_dir)

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

def main():
    files_to_ai: List[FileData] = []
    ai_shared_file_types = []

    run_claude = '-ai' in sys.argv
    run_readlast = '-readlast' in sys.argv
    force = '-f' in sys.argv
    
    if '-pdf' in sys.argv:
        ai_shared_file_types.append("pdf")
    
    if '-img' in sys.argv:
        files_to_ai = add_images(files_to_ai)
        ai_shared_file_types.append("img")
    
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

    files_to_ai = add_source(files_to_ai, SOURCE, ai_shared_file_types)
    tree_dirs = get_directory_tree(TREE_DIRS, files_to_ai)

    # Build the message content array with images and text
    message_content = []
    data_files = ""

    # Scan file_to_ai: add images, build file data
    for file_to_ai in files_to_ai:
        if not file_to_ai.ai_share:
            continue

        if file_to_ai.file_type == "image":
            message_content.extend([{
                'type': file_to_ai.file_type,
                'source': {
                    'type': file_to_ai.ai_data_converted_type,
                    'media_type': file_to_ai.media_type,
                    'data': file_to_ai.ai_data_converted
                },
            }])
        
        elif file_to_ai.file_type == "text":
            message_content.extend([{
                "type": "text", 
                "text": f"\n----- {file_to_ai.path_rel} -----\n{file_to_ai.ai_data_converted}-----\n\n"
            }])
            data_files += message_content[-1]["text"]
        
        elif file_to_ai.file_type == "bin":
            message_content.extend([{
                "type": "text", 
                "text": f"\n----- {file_to_ai.path_rel} -----\n{file_to_ai.ai_data_converted}-----\n\n"
            }])
            data_files += message_content[-1]["text"]
        else:
            print(f"Unexpected file type while building message_content: {file_to_ai.file_type}")
    
    # Add prompt
    message_content.extend([{"type": "text", "text": PROMPT}])

    # Add directory tree
    message_content.extend([{"type": "text", "text": f"Directory tree structure: {tree_dirs}"}])
    
    # Calculate token estimates
    print(f"Input tokens [ESTIMATED]: {(len(str(message_content))+len(str(SYSTEM))) // 4}")
    
    export_md_file("\n\n".join([SYSTEM, PROMPT, tree_dirs, data_files]), "userfullprompt.md")
    export_md_file(message_content, "message_content.md")
    
    # Check -ai flag
    if not run_claude:
        print("Not executing AI request...")
        return

    # Check for uncommitted git changes
    if not force and has_uncommitted_changes():
        print("ERROR: You have uncommitted changes in your git repository.")
        print("Please commit or stash your changes before running this script.")
        sys.exit(1)

    client = Anthropic(api_key=ANTHROPIC["API_KEY"])
    print("Sending request to Claude...")

    # Enable streaming
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

    # Ensure clauderesponse-recv.md is empty before streaming
    recv_path = os.path.join(output_dir, "clauderesponse-recv.md")
    with open(recv_path, "w", encoding="utf-8") as f:
        f.write("")

    for event in response:
        # Store raw event data
        raw_data.append({
            'type': event.type,
            'event': str(event)
        })
        
        try:
            if event.type == "content_block_delta":
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
                        
                        # Append chunk to clauderesponse-recv.md
                        with open(recv_path, "a", encoding="utf-8") as f:
                            f.write(chunk)

                else:
                    print(f"\n[DEBUG: Unexpected delta structure in {event.type}]")
            
            elif event.type == "message_start":
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
