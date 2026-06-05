"""
EVA's tools — auto-discovered from this package.

Convention for tool files:
    - @tool decorated function  →  collected directly
    - make_* factory function   →  called with action_buffer, result collected
"""

import importlib
import os
from langchain_core.tools import BaseTool

from config import logger
from eva.actions.action_buffer import ActionBuffer


class ToolError(Exception):
    """Raised by tools to carry their name alongside the error."""

    def __init__(self, message: str, tool_name: str):
        super().__init__(message)
        self.tool_name = tool_name


def handle_tool_error(e: Exception) -> str:
    """Translate tool errors into first-person messages I can reason about."""
    tool = getattr(e, "tool_name", None)
    prefix = f"My {tool} tool" if tool else "That tool"

    cause = e.__cause__ or e
    cause_cls = type(cause).__name__
    cause_str = str(cause).lower()

    if "Timeout" in cause_cls or "timeout" in cause_str:
        return f"{prefix} ran out of time — the service is slow or overloaded."

    if "Connection" in cause_cls or any(
        x in cause_str for x in ("connection", "unreachable", "socket")
    ):
        return f"{prefix} can't reach the service — there are connection issues."

    return f"{prefix} ran into an error: {cause}. It needs to be fixed."


def load_tools(action_buffer: ActionBuffer) -> list[BaseTool]:
    """Scan this folder, import each module, collect tools."""
    tools = []
    pkg_dir = os.path.dirname(__file__)

    for filename in sorted(os.listdir(pkg_dir)):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue

        module_name = filename[:-3]
        try:
            module = importlib.import_module(f".{module_name}", package=__package__)
        except Exception as e:
            logger.warning(f"Tools: skipped {module_name} — {e}")
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)

            if isinstance(obj, BaseTool):
                tools.append(obj)
                logger.debug(f"Tools: loaded {obj.name}")
            elif callable(obj) and attr_name.startswith("make_"):
                try:
                    tool = obj(action_buffer)
                    tools.append(tool)
                    logger.debug(f"Tools: loaded {tool.name} (factory)")  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(f"Tools: factory {attr_name} failed — {e}")

    return tools
