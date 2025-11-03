from warnings import filterwarnings

from pyfiglet import Figlet
from rich.console import Console
from rich.style import Style

filterwarnings("ignore", category=UserWarning)

APP_NAME: str = __package__

art = Figlet(font="dos_rebel", justify="center", width=110).renderText("VISUALIZR")

console = Console()
console.print(art, no_wrap=True, new_line_start=True)
console.print(
    "Video Generation with AniTalker Model.",
    style=Style(bold=True),
    no_wrap=True,
    justify="center",
)
console.print(
    "Developed by [link=https://github.com/AlphaSphereDotAI]AlphaSphere.AI[/link]",
    style=Style(bold=True),
    no_wrap=True,
    justify="center",
    end="\n\n\n\n",
)
