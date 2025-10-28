from gradio import Blocks

from visualizr.app.runner import app


def main() -> None:
    """Launch the Gradio voice generation web app."""
    application: Blocks = app.gui()
    application.queue(api_open=True).launch(
        mcp_server=True,
        show_api=True,
        enable_monitoring=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
