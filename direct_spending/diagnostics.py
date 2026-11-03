"""Optional storage plot artifact consumed by the current runner's saved-run view."""
from pathlib import Path


def write_storage_plot(rows, directory):
    import matplotlib.pyplot as plt
    traces = [(row["validation_name"], row["storage_diagnostics"])
              for row in rows if "storage_diagnostics" in row]
    if not traces:
        return None
    fig, ax = plt.subplots(figsize=(10, 3))
    for name, diagnostics in traces:
        for trace in diagnostics["storage_trace"]:
            ax.plot(trace, label=name)
    projected = any(row.get("certificate", {}).get("readout") == "projected" for row in rows)
    title = "Output projection + linear path" if projected else "Direct spending"
    ax.set(xlabel="Samples processed (including warm-up)", ylabel="Storage",
           title=title + " — storage per validation trajectory")
    ax.legend()
    fig.tight_layout()
    path = Path(directory) / "storage.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)
