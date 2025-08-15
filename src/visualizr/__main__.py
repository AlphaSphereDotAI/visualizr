from gradio import Blocks

from visualizr.gui import app_block


def main() -> None:
    """Launch the Gradio voice generation web app."""
    app: Blocks = app_block()
    app.queue(api_open=True).launch(
        server_port=7860,
        debug=True,
        mcp_server=True,
        show_api=True,
        enable_monitoring=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
