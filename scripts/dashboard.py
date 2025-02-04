import matplotlib.pyplot as plt
import json
import time
import threading
import os
import argparse
from flask import Flask, render_template_string, send_from_directory

# Ensure Matplotlib does not use any GUI backend
import matplotlib
matplotlib.use('Agg')

# Parse input arguments
parser = argparse.ArgumentParser(description="Monitor training metrics and visualize losses.")
parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing training_metrics.json")
args = parser.parse_args()

# File path
file_path = os.path.join(args.input_dir, "training_metrics.json")

# Ensure the static directory exists
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)
plot_path = os.path.join(static_dir, "loss_plot.png")

app = Flask(__name__)

# Extract loss-related keys
loss_keys = ["total_loss", "dino_local_crops_loss", "dino_global_crops_loss", "koleo_loss", "ibot_loss", "lr"]

data_store = {"iterations": [], "loss_values": {key: [] for key in loss_keys}}

def load_data():
    """Load data from JSON file (newline-delimited JSON)"""
    if not os.path.exists(file_path):
        return
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    data_store["iterations"] = [entry["iteration"] for entry in data]
    for key in loss_keys:
        data_store["loss_values"][key] = [entry[key] for entry in data]

def update_plot():
    """Continuously update the plot every 10 seconds"""
    while True:
        load_data()
        generate_plot()
        time.sleep(10)

def generate_plot():
    """Generate individual plots for each loss key"""
    if not data_store["iterations"]:
        return  # Skip if no data
    
    fig, axes = plt.subplots(len(loss_keys), 1, figsize=(10, 12), sharex=True)
    for ax, key in zip(axes, loss_keys):
        if data_store["loss_values"][key]:
            ax.plot(data_store["iterations"], data_store["loss_values"][key], label=key)
        ax.set_ylabel(key)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlabel("Iteration")
    plt.suptitle("Loss Trends Over Iterations (Individual Scales)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_path)
    plt.close()

@app.route("/")
def index():
    if not os.path.exists(plot_path):
        return "<h1>No plot available yet. Waiting for data...</h1>"
    return render_template_string("""
    <html>
        <head><title>Loss Trends</title></head>
        <body>
            <h1>Loss Trends Over Iterations</h1>
            <img src="/plot" alt="Loss Plot" width="800">
            <meta http-equiv="refresh" content="10">
        </body>
    </html>
    """)

@app.route("/plot")
def serve_plot():
    if not os.path.exists(plot_path):
        return "Plot not found", 404
    return send_from_directory(static_dir, "loss_plot.png")

if __name__ == "__main__":
    threading.Thread(target=update_plot, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
