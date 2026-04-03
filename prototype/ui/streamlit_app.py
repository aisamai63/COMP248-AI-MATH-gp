"""Compatibility Streamlit entrypoint under the ui module.

This wrapper preserves existing behavior by delegating to prototype.app.main.
"""

from prototype.app import main


if __name__ == "__main__":
    main()
