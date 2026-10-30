"""These programs are input only. `best_strategy` is measured by the labeler."""

from __future__ import annotations

from pathlib import Path
import re

from tac.parser import parse_tac
from tac.program import TACProgram


SAMPLE_FILE = Path(__file__).resolve().parents[1] / "examples" / "sample_programs.txt"
_HEADER = re.compile(r"^===\s*(.+?)\s*===\s*$")


def load_sample_sources(path: Path | None = None) -> list[tuple[str, str]]:
    """Return (program_id, tac_text) in file order."""
    sample_path = path or SAMPLE_FILE
    programs: list[tuple[str, str]] = []
    current_id: str | None = None
    current_lines: list[str] = []

    for line in sample_path.read_text(encoding="utf-8").splitlines():
        header = _HEADER.match(line)
        if header:
            if current_id is not None:
                programs.append((current_id, "\n".join(current_lines).strip("\n")))
            current_id = header.group(1)
            current_lines = []
            continue
        if current_id is not None:
            current_lines.append(line)

    if current_id is not None:
        programs.append((current_id, "\n".join(current_lines).strip("\n")))
    return programs


def load_sample_programs(path: Path | None = None) -> list[tuple[str, TACProgram]]:
    """Parse every hand-written sample."""
    return [
        (program_id, parse_tac(source))
        for program_id, source in load_sample_sources(path)
    ]
