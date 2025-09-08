from logging import INFO, basicConfig, getLogger
from warnings import filterwarnings

from pyfiglet import Figlet
from rich.console import Console
from rich.logging import RichHandler
from rich.style import Style

filterwarnings("ignore", category=UserWarning)

figlet = Figlet(font="dos_rebel", width=110)
art = figlet.renderText("VISUALIZR")

console = Console()
console.print(art, style=Style(frame=True), new_line_start=True)
console.print(
    "Video Generation with AniTalker Model.",
    style=Style(bold=True),
    no_wrap=True,
    justify="center",
)
console.print(
    "Developed by (https://github.com/AlphaSphereDotAI)",
    style=Style(bold=True),
    no_wrap=True,
    justify="center",
)

basicConfig(
    level=INFO,
    handlers=[RichHandler(level=INFO, console=console, rich_tracebacks=True)],
    format="%(name)s | %(process)d | %(message)s",
)
logger = getLogger(__name__)
logger.disabled = True
