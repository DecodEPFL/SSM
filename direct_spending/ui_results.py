"""Saved-run views inside the current benchmark window, using its standard reports/plots."""
from __future__ import annotations

import json
from pathlib import Path

import tkinter as tk
from tkinter import ttk


class RunResults:
    def __init__(self, notebook):
        self.notebook, self.records, self.report_path = notebook, [], None
        self.results = ttk.Frame(notebook, padding=5)
        self.details = ttk.Frame(notebook, padding=5)
        self.plots = ttk.Frame(notebook, padding=5)
        notebook.add(self.results, text="Results / saved runs")
        notebook.add(self.details, text="Run details / storage")
        notebook.add(self.plots, text="Prediction and loss plots")
        columns = ("benchmark", "model", "rmse", "fit", "best_val_loss", "status")
        self.table = ttk.Treeview(self.results, columns=columns, show="headings", height=7)
        for key in columns:
            self.table.heading(key, text=key.replace("_", " "))
            self.table.column(key, width=130 if key != "model" else 320, stretch=True)
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self._select)
        self.text = tk.Text(self.details, wrap="word", height=9)
        self.text.pack(fill="both", expand=True)
        self.plot = ttk.Label(self.plots, text="Select a completed run to see its existing training plot.")
        self.plot.pack(fill="both", expand=True)
        self.image = None

    def load(self, path):
        path = Path(path)
        data = json.loads(path.read_text())
        if not isinstance(data.get("results"), list):
            raise ValueError("Choose a nonlinear benchmark report.json file")
        self.records, self.report_path = data["results"], path
        for row in self.table.get_children():
            self.table.delete(row)
        for index, row in enumerate(self.records):
            values = (row.get("benchmark"), row.get("model_label", row.get("model")),
                      self._number(row.get("rmse")), self._number(row.get("fit")),
                      self._number(row.get("best_val_loss")), row.get("status"))
            self.table.insert("", "end", iid=str(index), values=values)
        if self.records:
            self.table.selection_set("0")
            self._select()
        self.notebook.select(self.results)

    @staticmethod
    def _number(value):
        return f"{value:.5g}" if isinstance(value, (int, float)) else "—"

    def _select(self, _event=None):
        selected = self.table.selection()
        if not selected:
            return
        record = self.records[int(selected[0])]
        # Keep full traces in the normal report; render their summaries in details.
        view = dict(record)
        summaries = []
        for sequence in record.get("validation_sequences", []):
            item = dict(sequence)
            if "storage_diagnostics" in item:
                diag = dict(item["storage_diagnostics"])
                trace = diag.pop("storage_trace", None)
                if trace is not None:
                    diag["storage_trace_samples"] = [len(row) for row in trace]
                norms = diag.pop("raw_proposal_norm", None)
                if norms is not None:
                    diag["raw_proposal_norm_samples"] = [len(row) for row in norms]
                item["storage_diagnostics"] = diag
            summaries.append(item)
        view["validation_sequences"] = summaries
        view.pop("history", None)
        heading = ""
        if record.get("model_family") == "energy_projection":
            heading = "Clipping: the combined NN + learned linear proposal exceeded available energy.\n\n"
        elif record.get("model_family") == "direct_spending":
            heading = "Head saturation is ||z|| > 1; it is not physical output clipping.\n\n"
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", heading + json.dumps(view, indent=2))
        self.text.configure(state="disabled")
        artifact = record.get("artifacts", {}).get("figure")
        if artifact and Path(artifact).is_file():
            from PIL import Image, ImageTk
            image = Image.open(artifact)
            image.thumbnail((1000, 600))
            self.image = ImageTk.PhotoImage(image)
            self.plot.configure(image=self.image, text="")
        else:
            self.plot.configure(image="", text="This saved run has no prediction/loss plot artifact.")

    def show_storage_trace(self):
        """Select a saved trace through the existing plot view; no separate dashboard."""
        selected = self.table.selection()
        if not selected:
            return
        record = self.records[int(selected[0])]
        artifact = record.get("artifacts", {}).get("storage_figure")
        if artifact and Path(artifact).is_file():
            from PIL import Image, ImageTk
            image = Image.open(artifact)
            image.thumbnail((1000, 600))
            self.image = ImageTk.PhotoImage(image)
            self.plot.configure(image=self.image, text="")
            self.notebook.select(self.plots)
        else:
            self.plot.configure(image="", text="Enable Storage diagnostics and run again to save a storage trace.")
            self.notebook.select(self.plots)
