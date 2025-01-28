import matplotlib.pyplot as plt
import json
import time
import threading
import os
import argparse
from flask import Flask, render_template_string

# Parse input arguments
parser = argparse.ArgumentParser(description="Monitor training metrics and visualize losses.")
parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing training_metrics.json")
args = parser.parse_args()

# File path
file_path = os.path.join(args.input_dir, "training_metrics.json")

app = Flask(__name__)

# Extract loss-related keys
loss_keys = ["total_loss", "dino_local_crops_loss", "dino_global_crops_loss", "koleo_loss", "ibot_loss"]

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
        time.sleep(10)

def generate_plot():
    """Generate plot as an HTML page"""
    plt.figure(figsize=(10, 8))
    for key in loss_keys:
        plt.plot(data_store["iterations"], data_store["loss_values"][key], label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Loss Value")
    plt.title("Loss Trends Over Iterations")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("static/loss_plot.png")
    plt.close()

@app.route("/")
def index():
    generate_plot()
    return render_template_string("""
    <html>
        <head><title>Loss Trends</title></head>
        <body>
            <h1>Loss Trends Over Iterations</h1>
            <img src="/static/loss_plot.png" alt="Loss Plot" width="800">
            <meta http-equiv="refresh" content="10">
        </body>
    </html>
    """)

if __name__ == "__main__":
    threading.Thread(target=update_plot, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
