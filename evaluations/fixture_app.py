"""Disposable GUI oracle. Reads/writes only its explicitly supplied test directory."""
import json
import os
from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel


def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("WisprHelp Evaluation Fixture")
    layout = QVBoxLayout(window)
    editor = QLineEdit()
    editor.setAccessibleName("Evaluation Editor")
    editor.setObjectName("evaluation_editor")
    button = QPushButton("Next Test Page")
    button.setAccessibleName("Next Test Page")
    status = QLabel("Page 1")
    layout.addWidget(editor)
    layout.addWidget(button)
    layout.addWidget(status)
    clicks = [0]

    def publish():
        data = {"pid": os.getpid(), "text": editor.text(), "page": status.text(), "clicks": clicks[0]}
        temp = root / "oracle.tmp"
        temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temp.replace(root / "oracle.json")

    def next_page():
        clicks[0] += 1
        status.setText("Page 2")
        publish()

    button.clicked.connect(next_page)
    editor.textChanged.connect(publish)
    timer = QTimer()
    timer.timeout.connect(lambda: app.quit() if (root / "stop").exists() else None)
    timer.start(50)
    QTimer.singleShot(120_000, app.quit)  # Fixture cannot outlive its bounded run.
    window.resize(480, 160)
    window.show()
    window.activateWindow()
    editor.setFocus()
    publish()
    app.exec()


if __name__ == "__main__":
    main()
