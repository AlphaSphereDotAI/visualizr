from argparse import ArgumentParser, Namespace

from gradio import Blocks

from visualizr import DEBUG, SERVER_NAME, SERVER_PORT
from visualizr.gui import app_block


def main() -> None:
    """Launch the Gradio voice generation web application."""
    parser = ArgumentParser(description="EchoMimic")
    parser.add_argument(
        "--server_name", type=str, default="localhost", help="Server name"
    )
    parser.add_argument("--server_port", type=int, default=3001, help="Server port")
    args: Namespace = parser.parse_args()
    app: Blocks = app_block()
    app.queue(api_open=True).launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        debug=DEBUG,
        mcp_server=True,
        show_api=True,
        enable_monitoring=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
