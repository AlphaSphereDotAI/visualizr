from logging import INFO, WARNING, basicConfig, getLogger

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
logger = getLogger(__package__)
