import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def slugify(value: str, default: str = "run") -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = value.strip("-._")
    return value or default


def make_run_dir(output_root: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-{slugify(experiment_name)}"
    run_dir = Path(output_root) / run_name
    for child in ("samples", "figures", "checkpoints", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_metadata(run_dir: Path, metadata: Dict[str, Any], filename: str = "run_meta.json") -> Path:
    payload = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
    }
    path = run_dir / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def module_hparams(module: Any) -> Dict[str, Any]:
    hparams: Dict[str, Any] = {}
    for key, value in vars(module).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            hparams[key] = value
    return hparams


def maybe_write_summary(run_dir: Optional[Path], lines: Dict[str, Any]) -> Optional[Path]:
    if run_dir is None:
        return None
    path = run_dir / "summary.txt"
    text = "\n".join(f"{key}: {value}" for key, value in lines.items())
    path.write_text(text + "\n", encoding="utf-8")
    return path
