from warnings import filterwarnings

from pyfiglet import Figlet
from rich.console import Console
from rich.panel import Panel
from rich.style import Style

filterwarnings("ignore", category=UserWarning)

art = Figlet(font="dos_rebel", justify="center").renderText("VISUALIZR")

console = Console()
console.print(Panel(art))

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
