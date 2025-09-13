from warnings import filterwarnings

from pyfiglet import Figlet
from rich.console import Console
from rich.panel import Panel

filterwarnings("ignore", category=UserWarning)

art = Figlet(font="slant", justify="center").renderText("VISUALIZR")

console = Console()
console.print(Panel(art))
