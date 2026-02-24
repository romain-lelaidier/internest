import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np


# Function to update the plot
def update_plots():
    ax1.clear()
    ax1.scatter(df['time'], df['x'], c='blue')
    ax1.set_title('X')

    ax2.clear()
    ax2.plot(df['time'], df['z'], c='red')
    ax2.set_title('Z')

    canvas.draw()

# Function to update the text zone
def update_text_zone(text_widget, data):
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, data.to_string())
    text_widget.config(state=tk.DISABLED)


# Event handler for file changes
class CSVHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global df
        if event.src_path.endswith('.csv'):
            new_df = pd.read_csv(event.src_path)
            if not new_df.equals(df):
                df = new_df
                update_plots()
                update_text_zone(text_zone, df["birdnet_id"])



if __name__ == '__main__':
    # Global DataFrame to store the data
    df = pd.DataFrame(columns=['time', 'x', 'y', 'z', 'cost', 'id', 'confidence', 'start_time', 'end_time', 'birdnet_id'])

    # Set up the main window
    root = tk.Tk()
    root.title("Real-Time Dashboard")
    # Text zone
    text_frame = ttk.Frame(root)
    text_frame.pack(fill=tk.BOTH, expand=True)
    text_zone = tk.Text(text_frame, height=10)
    text_zone.pack(fill=tk.BOTH, expand=True)

    # Plots
    plot_frame = ttk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Set up the observer
    observer = Observer()
    observer.schedule(CSVHandler(), path='.', recursive=False)
    observer.start()

    # Start the GUI
    root.mainloop()

    # Cleanup
    observer.stop()
    observer.join()
