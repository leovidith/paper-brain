import os
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import filedialog

import config

from chat import controller


def main():
    root = tk.Tk()
    root.withdraw()

    pdf_paths = filedialog.askopenfilenames(
        title="Select PDF files",
        filetypes=[("PDF files", "*.pdf")]
    )

    if not pdf_paths:
        print("[ERROR] No files selected.")
        return

    print(f"Selected {len(pdf_paths)} PDF(s): {[os.path.basename(p) for p in pdf_paths]}")

    controller(list(pdf_paths))


if __name__ == "__main__":
    main()