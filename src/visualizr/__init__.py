from pyfiglet import Figlet
from rich.console import Console
from rich.style import Style

figlet = Figlet(font="slant", justify="center")
art = figlet.renderText("Visualizr")

console = Console()
console.print(art, style=Style(bold=True))
