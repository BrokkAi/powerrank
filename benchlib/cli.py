import json
import os
import pathlib
import re
import subprocess
import sys
import threading
from typing import List

CLI_BIN: pathlib.Path | None = None

# Environment variables that contain direct directory paths
CACHE_DIR_VARS = [
    "BRK_WORKTREE_ROOT",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "GOCACHE",
    "GOMODCACHE",
    "GOTMPDIR",
    "GRADLE_USER_HOME",
    "npm_config_cache",
    "PIP_CACHE_DIR",
    "NUGET_PACKAGES",
    "DOTNET_CLI_HOME",
    "TMPDIR",
]

# Environment variables with embedded paths (need parsing)
EMBEDDED_PATH_VARS = {
    "MAVEN_OPTS": r"-Dmaven\.repo\.local=(\S+)",
    "SBT_OPTS": r"-Dsbt\.global\.base=(\S+)|-Dsbt\.ivy\.home=(\S+)",
}

def set_cli_bin(path: pathlib.Path) -> None:
    """Set the CLI binary path. If the path is a directory, appends 'cli'."""
    global CLI_BIN
    if path.is_dir():
        CLI_BIN = path / "cli"
    else:
        CLI_BIN = path


def validate_api_key() -> None:
    """
    Validate that BROKK_API_KEY is present in environment or config file.
    If found in config but not env, it sets the environment variable.
    Exits with error if not found.
    """
    if os.getenv("BROKK_API_KEY"):
        return

    props_path = pathlib.Path.home() / ".config" / "brokk" / "brokk.properties"
    if props_path.exists():
        with open(props_path, "r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if stripped.startswith("brokkApiKey="):
                    key = stripped.split("=", 1)[1].strip()
                    if key:
                        os.environ["BROKK_API_KEY"] = key
                        return

    print("Error: BROKK_API_KEY environment variable is required.", file=sys.stderr)
    print("Alternatively, set brokkApiKey in ~/.config/brokk/brokk.properties", file=sys.stderr)
    print("The Brokk CLI requires this API key to authenticate with Brokk services.", file=sys.stderr)
    sys.exit(1)


def validate_cache_dirs() -> None:
    """
    Validate and create cache directories from environment variables.
    Extracts paths from CACHE_DIR_VARS (direct paths) and EMBEDDED_PATH_VARS
    (paths embedded in JVM-style options), then creates them if missing.
    Exits with clear error if creation fails.
    """
    paths_to_create: List[tuple[str, str]] = []  # (var_name, path)

    # Direct path variables
    for var in CACHE_DIR_VARS:
        value = os.getenv(var)
        if value:
            paths_to_create.append((var, value))

    # Embedded path variables (parse with regex)
    for var, pattern in EMBEDDED_PATH_VARS.items():
        value = os.getenv(var)
        if value:
            for match in re.finditer(pattern, value):
                for group in match.groups():
                    if group:
                        paths_to_create.append((var, group))

    # Create directories
    for var_name, path in paths_to_create:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"Error: Cannot create cache directory for {var_name}", file=sys.stderr)
            print(f"  Path: {path}", file=sys.stderr)
            print(f"  Reason: {e}", file=sys.stderr)
            sys.exit(1)


def _get_cli_version() -> str:
    try:
        result = subprocess.run(
            [str(CLI_BIN), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _get_cli_models_json() -> list:
    env_val = os.getenv("BROKK_FAVORITE_MODELS")
    if env_val:
        try:
            return json.loads(env_val)
        except json.JSONDecodeError:
            print(f"Error: Cannot parse BROKK_FAVORITE_MODELS: {env_val}", file=sys.stderr)
            sys.exit(1)

    try:
        result = subprocess.run(
            [str(CLI_BIN), "--list-models"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Error: Failed to run CLI for model validation: {e}", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(f"Error: CLI --list-models failed (exit {result.returncode}):", file=sys.stderr)
        output = (result.stdout or "") + (result.stderr or "")
        print(output.strip(), file=sys.stderr)
        sys.exit(1)

    stdout_lines = result.stdout.strip().splitlines()
    json_line = stdout_lines[-1] if stdout_lines else ""
    try:
        return json.loads(json_line)
    except json.JSONDecodeError:
        print(f"Error: Cannot parse --list-models output: {json_line}", file=sys.stderr)
        sys.exit(1)


def _read_proxy_setting() -> str:
    env_val = os.getenv("BROKK_PROXY")
    if env_val:
        return env_val
    props_path = pathlib.Path.home() / ".config" / "brokk" / "brokk.properties"
    if props_path.exists():
        with open(props_path, "r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if stripped.startswith("llmProxySetting="):
                    return stripped.split("=", 1)[1].strip()
    return "BROKK"


_cli_info_cache: dict | None = None
_cli_info_lock = threading.Lock()


def get_cli_info() -> dict:
    global _cli_info_cache
    if _cli_info_cache is not None:
        return _cli_info_cache
    with _cli_info_lock:
        if _cli_info_cache is not None:
            return _cli_info_cache
        models_list = _get_cli_models_json()
        _cli_info_cache = {
            "cliVersion": _get_cli_version(),
            "proxy": _read_proxy_setting(),
            "favoriteModels": models_list,
        }
        return _cli_info_cache


def validate_models(models: List[str]) -> None:
    info = get_cli_info()
    try:
        available = {e["alias"] for e in info["favoriteModels"]}
    except KeyError:
        print("Error: Cannot parse model aliases from CLI output", file=sys.stderr)
        sys.exit(1)

    invalid = [m for m in models if m not in available]
    if invalid:
        print(f"Error: Unknown model alias(es): {', '.join(invalid)}", file=sys.stderr)
        print(f"Available aliases: {', '.join(sorted(available))}", file=sys.stderr)
        sys.exit(1)


def run_cli(cmd: List[str], log_file: pathlib.Path, env: dict | None = None) -> subprocess.CompletedProcess:
    """
    Execute a Brokk CLI command and append output to the supplied log-file.
    If BB_DEBUG is set, output is also echoed to stderr.
    """
    debug = bool(os.getenv("BB_DEBUG"))
    if debug:
        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        if env and "JAVA_HOME" in env:
            print(f"Using JAVA_HOME: {env['JAVA_HOME']}", file=sys.stderr)

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    ) as proc, open(log_file, "ab") as log_fp:
        for line in proc.stdout:
            log_fp.write(line)
            if debug:
                try:
                    sys.stderr.buffer.write(line)
                    sys.stderr.flush()
                except AttributeError:
                    sys.stderr.write(line.decode(errors="replace"))
                    sys.stderr.flush()
        proc.wait()

    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=None, stderr=None)
