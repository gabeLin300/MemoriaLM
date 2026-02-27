import os
from pathlib import Path


def load_local_env(filename: str = ".env.local") -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / filename
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        # Keep shell/env precedence if already set.
        if key not in os.environ:
            os.environ[key] = value
