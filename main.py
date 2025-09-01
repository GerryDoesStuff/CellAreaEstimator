from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from PyQt6.QtWidgets import QApplication

from gui import MainWindow

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cell Area Estimator")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    args = parser.parse_args(argv)
    level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.basicConfig(level=level)
    logger.debug("Starting application with log level %s", args.log_level)
    app = QApplication(list(argv) if argv is not None else sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
