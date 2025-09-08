from logging import INFO, basicConfig, getLogger
from warnings import filterwarnings

from pyfiglet import Figlet
from rich.console import Console
from rich.logging import RichHandler
from rich.style import Style

filterwarnings("ignore", category=UserWarning)

figlet = Figlet(font="slant", justify="center")
art = figlet.renderText("VISUALIZR")

console = Console()
console.print(art, style=Style(bold=True))

basicConfig(
    level=INFO,
    handlers=[RichHandler(level=INFO, console=console, rich_tracebacks=True)],
    format="%(name)s: %(asctime)s | %(process)d | %(message)s",
)
logger = getLogger(__name__)
logger.disabled = True
