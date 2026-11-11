"""Read JSONL, gzip, or one selected JSONL member inside a ZIP."""
import ast
from contextlib import contextmanager
import gzip
import io
from pathlib import Path
import zipfile


@contextmanager
def open_text(spec):
    archive, separator, member = str(spec).partition("::")
    if separator:
        with zipfile.ZipFile(archive) as zf, zf.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as stream:
                yield stream
    elif Path(archive).suffix == ".gz":
        with gzip.open(archive, "rt", encoding="utf-8") as stream:
            yield stream
    else:
        with open(archive, encoding="utf-8") as stream:
            yield stream


def records(path):
    import json
    with open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = ast.literal_eval(line)  # Amazon 2014 metadata can be Python literals.
                if not isinstance(row, dict):
                    raise ValueError("Expected object")
                yield row
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid record at {path}:{line_number}") from e


def text_value(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {text_value(v)}" for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return ", ".join(text_value(v) for v in value if v is not None)
    return str(value).strip()

