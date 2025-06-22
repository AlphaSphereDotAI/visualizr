from os import getenv

from gradio import Blocks

from visualizr.gui import app_block


def main() -> None:
    """Launch the Gradio voice generation web application."""
    app: Blocks = app_block()
    app.queue(api_open=True).launch(
        server_name=getenv(key="GRADIO_SERVER_NAME", default="localhost"),
        server_port=int(getenv(key="GRADIO_SERVER_PORT", default="8080")),
        debug=getenv(key="GRADIO_DEBUG", default="1"),
        mcp_server=True,
        show_api=True,
        enable_monitoring=True,
        show_error=True,
        app_kwargs={"docs_url": "/docs"}
    )


if __name__ == "__main__":
    main()
