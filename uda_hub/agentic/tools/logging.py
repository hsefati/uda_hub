import logging
from typing import Any


# Shared structured logger for UDA-Hub agent nodes
logger = logging.getLogger("uda_hub.nodes")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _format_kv(**kwargs: Any) -> str:
    parts = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, float):
            s = f"{v:.2f}"
        else:
            s = str(v)
        if " " in s:
            s = f'"{s}"'
        parts.append(f"{k}={s}")
    return " ".join(parts)


def node_log(node: str, **kwargs: Any) -> None:
    """Emit a single-line structured log like: [CLASSIFIER] key=val key2=val2

    Keep logs simple and grep-friendly (not JSON).
    """
    tag = node.upper()
    logger.info(f"[{tag}] {_format_kv(**kwargs)}")
