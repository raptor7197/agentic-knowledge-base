import os
import subprocess
from typing import Optional

current_dir = os.getcwd()

# Caches
file_cache = {}
dir_cache = {}

def change_directory(path: str) -> str:
    """Change the current working directory for subsequent operations."""
    global current_dir
    full_path = os.path.abspath(path)
    if os.path.isdir(full_path):
        current_dir = full_path
        return f"Changed directory to {current_dir}"
    else:
        return f"Directory {full_path} does not exist"

def read_file(file_path: str) -> str:
    """Read the contents of a file given its absolute path."""
    try:
        mtime = os.path.getmtime(file_path)
        key = (file_path, mtime)
        if key in file_cache:
            return file_cache[key]
        with open(file_path, 'r') as f:
            content = f.read()
        file_cache[key] = content
        return content
    except Exception as e:
        return f"Error reading file: {e}"

def search_code(pattern: str, path: Optional[str] = None) -> str:
    """Search for a regex pattern in files within the given path."""
    if path is None:
        path = current_dir
    try:
        result = subprocess.run(['grep', '-r', pattern, path], capture_output=True, text=True)
        return result.stdout or "No matches found."
    except Exception as e:
        return f"Error searching: {e}"

def list_directory(path: Optional[str] = None) -> str:
    """List files and directories in the given path."""
    if path is None:
        path = current_dir
    try:
        mtime = os.path.getmtime(path)
        key = (path, mtime)
        if key in dir_cache:
            return dir_cache[key]
        listing = "\n".join(os.listdir(path))
        dir_cache[key] = listing
        return listing
    except Exception as e:
        return f"Error listing directory: {e}"

def run_command(command: str) -> str:
    """Run a bash command and return the output."""
    try:
        # Execute in the current directory
        full_command = f"cd '{current_dir}' && {command}"
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running command: {e}"