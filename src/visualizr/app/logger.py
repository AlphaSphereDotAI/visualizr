from logging import INFO, WARNING, basicConfig, getLogger
from typing import Literal

from gradio import Error, Info, Warning as grWarning
from rich.logging import RichHandler

from visualizr import console

basicConfig(
    level=INFO,
    handlers=[
        RichHandler(
            level=INFO,
            console=console,
            rich_tracebacks=True,
        ),
    ],
    format="%(name)s | %(process)d | %(message)s",
)
getLogger("httpx").setLevel(WARNING)
logger = getLogger(__name__)


def log_message_with_ui(
    message: str,
    level: Literal[
        "critical",
        "error",
        "warning",
        "info",
        "debug",
        "exception",
    ] = "info",
    *,
    ui: bool = True,
) -> None:
    """Log a message and displays it in the Gradio UI."""
    match level:
        case "critical":
            logger.critical(message)
        case "error":
            logger.error(message)
        case "exception":
            logger.exception(message)
        case "warning":
            logger.warning(message)
        case "info":
            logger.info(message)
        case "debug":
            logger.debug(message)
    if ui:
        match level:
            case "critical" | "error" | "exception":
                Error(message)
            case "warning":
                grWarning(message)
            case "info" | "debug":
                Info(message)
