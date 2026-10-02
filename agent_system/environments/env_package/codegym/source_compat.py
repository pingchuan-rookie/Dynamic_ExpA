"""Explicit source compatibility for generated CodeGym environments, without rewriting data."""
from pathlib import Path


_MIN_PATH_COST = "Codeforces_5537_I__MinPathCostEnv.py"
_BROKEN_COLUMN = "[{self.dp[i][0] for i in range(self.n)}]"
_FIXED_COLUMN = "{[self.dp[i][0] for i in range(self.n)]}"
_COLUMN_MESSAGES = (
    "First column does not need filling, current state: ",
    "First column filled, current state: ",
)


def compatible_source(path: str | Path) -> tuple[str, tuple[str, ...]]:
    """Repair only the two known observation expressions; unknown defects still fail compilation."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    repairs = []
    if path.name == _MIN_PATH_COST:
        for message in _COLUMN_MESSAGES:
            old = f'return f"{message}{_BROKEN_COLUMN}"'
            new = f'return f"{message}{_FIXED_COLUMN}"'
            if source.count(old) == 1:
                source = source.replace(old, new, 1)
                repairs.append("min_path_cost_column_fstring")
    return source, tuple(repairs)


def compile_environment(path: str | Path):
    """Use identical source rules for runtime imports and preflight checks."""
    source, repairs = compatible_source(path)
    return compile(source, str(path), "exec"), repairs


def validate_environment_sources(directory: str | Path) -> dict:
    """Compile the complete source set without executing or modifying environment files."""
    files = sorted(Path(directory).glob("*Env.py"))
    if not files:
        raise RuntimeError(f"no *Env.py under {directory}")
    repaired, failures = {}, []
    for path in files:
        try:
            _, repairs = compile_environment(path)
        except (SyntaxError, UnicodeError) as exc:
            failures.append(f"{path.name}:{getattr(exc, 'lineno', '?')}: {exc}")
        else:
            if repairs:
                repaired[path.name] = list(repairs)
    if failures:
        raise RuntimeError(
            f"CodeGym source preflight failed: {len(failures)}/{len(files)} invalid environments; "
            + "; ".join(failures[:10])
        )
    return {"checked": len(files), "repairs": repaired}
