import tkinter as tk
import sys
from matplotlib import pyplot as plt

from perceptron_gui import MLPApp

if __name__ == "__main__":
    root = tk.Tk()
    app = MLPApp(root)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    app._redirect_stdout()

    def on_closing():
        if hasattr(sys.stdout, 'flush'):
            try:
                sys.stdout.flush()
            except Exception as e_flush:
                original_stdout.write(f"Error flushing GUI stdout: {e_flush}\n")

        if hasattr(app, 'history_plot_window') and app.history_plot_window:
            try:
                if app.history_plot_window.winfo_exists():
                    app.history_plot_window.destroy()
                app.history_plot_window = None
            except tk.TclError:
                original_stdout.write("TclError while trying to destroy plot window.\n")
            except Exception as e_plot_destroy:
                original_stdout.write(f"Error destroying plot window: {e_plot_destroy}\n")

        try:
            plt.close('all')
        except Exception as e_plt_close:
            original_stdout.write(f"Error closing all matplotlib plots: {e_plt_close}\n")

        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print("Restoring stdout/stderr and closing the application.")
        root.quit()

        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        root.mainloop()
    finally:
        if sys.stdout != original_stdout:
            sys.stdout = original_stdout
            print("stdout restored in finally block.")
        if sys.stderr != original_stderr:
            sys.stderr = original_stderr
            print("stderr restored in finally block.")
        print("Application finished.")