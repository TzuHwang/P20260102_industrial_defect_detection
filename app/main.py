"""Entry point for the defect-detection GUI.

    python -m app.main
"""

import sys

from PySide6.QtWidgets import QApplication

from app.defect_app.gui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
