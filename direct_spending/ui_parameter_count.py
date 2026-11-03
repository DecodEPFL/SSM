"""Responsive, on-demand parameter-count dialog; keep torch out of the UI process."""
import os
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import ttk


def show_parameter_count(root, command, repository):
    window = tk.Toplevel(root)
    window.title("Model parameter counts")
    window.geometry("730x400")
    ttk.Label(window, text="Snapshot of the current selection. Includes parameter matching.").pack(anchor="w", padx=10, pady=8)
    body = ttk.Frame(window)
    body.pack(fill="both", expand=True, padx=10)
    text = tk.Text(body, wrap="word", height=12)
    scrollbar = ttk.Scrollbar(body, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    text.insert("end", "Loading selected dataset dimensions and counting models…\nNo training is run.\n")
    text.configure(state="disabled")
    process, callback = None, None
    stream = tempfile.TemporaryFile(mode="w+t")
    cleaned = False

    def cleanup():
        nonlocal callback, cleaned
        if cleaned:
            return
        cleaned = True
        if callback is not None:
            window.after_cancel(callback)
            callback = None
        if process is not None and process.poll() is None:
            process.terminate()
            threading.Thread(target=process.wait, daemon=True).start()
        stream.close()

    def close():
        cleanup()
        window.destroy()

    ttk.Button(window, text="Close", command=close).pack(pady=8)
    window.protocol("WM_DELETE_WINDOW", close)

    def destroyed(event):
        if event.widget == window:
            cleanup()
    window.bind("<Destroy>", destroyed)

    def replace(message):
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("end", message)
        text.configure(state="disabled")

    try:
        process = subprocess.Popen([command[0], "-u", "-m", "direct_spending.parameter_count", *command[3:]],
            cwd=str(repository), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            text=True, stdout=stream, stderr=subprocess.STDOUT)
    except OSError as exc:
        stream.close()
        replace(f"Could not count parameters: {exc}")
        return window

    def poll():
        nonlocal callback
        callback = None
        if process.poll() is None:
            callback = window.after(100, poll)
            return
        stream.seek(0)
        output = stream.read()
        stream.close()
        replace(output if process.returncode == 0 else "Could not count parameters:\n" + output)
    callback = window.after(100, poll)
    return window
