import pathlib
import subprocess
import sys


def git_run(root: "pathlib.Path | str", *git_args: str):
    """
    Run a git command inside the given directory and return stdout stripped.
    Raises CalledProcessError on failure.
    """
    try:
        result = subprocess.run(
            ["git", *git_args],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
            timeout=300,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd} in {root}\n{e.stderr or e.output}", file=sys.stderr)
        raise
