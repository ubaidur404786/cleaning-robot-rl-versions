"""
Pre-run dialog that pops up before training starts.
Uses tkinter which ships with Python — no extra install needed.

The window lets you pick:
  - Mode:     Train / Test / Compare All
  - Agent:    q_learning / sarsa / dqn
  - Render:   ON / OFF  (opens a pygame window during episodes)
  - Episodes: how many episodes to run
"""

import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass


@dataclass
class RunConfig:
    """Holds every setting the user chose in the launch dialog."""
    mode: str = "train"        # "train", "test", or "compare"
    agent: str = "q_learning"  # "q_learning", "sarsa", "dqn"
    render: bool = False
    episodes: int = 500
    confirmed: bool = False    # False = user closed the window without clicking Start


def show_launcher() -> RunConfig:
    """
    Opens the settings popup.
    Blocks until the user clicks Start or Cancel / closes the window.
    Returns the filled-in RunConfig.
    """
    config = RunConfig()

    root = tk.Tk()
    root.title("Cleaning Robot RL — Settings")
    root.resizable(False, False)
    root.geometry("400x400+500+200")
    root.configure(bg="#1e1e2e")

    # ----- small helpers -------------------------------------------------

    def lbl(parent, text, bold=False):
        font = ("Segoe UI", 10, "bold") if bold else ("Segoe UI", 10)
        return tk.Label(parent, text=text, font=font, bg="#1e1e2e", fg="#cdd6f4")

    def section(parent, text):
        """Bold heading line with a bit of top padding."""
        lbl(parent, text, bold=True).pack(anchor="w", padx=20, pady=(14, 2))

    # ── Mode ─────────────────────────────────────────────────────────────
    section(root, "Mode")
    mode_var = tk.StringVar(value="train")
    row = tk.Frame(root, bg="#1e1e2e")
    row.pack(anchor="w", padx=30)
    for label, val in [("Train", "train"), ("Test", "test"), ("Compare All", "compare")]:
        tk.Radiobutton(
            row, text=label, variable=mode_var, value=val,
            bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
            activebackground="#1e1e2e", font=("Segoe UI", 10),
        ).pack(side="left", padx=6)

    # ── Agent ─────────────────────────────────────────────────────────────
    section(root, "Agent  (ignored in Compare mode)")
    agent_var = tk.StringVar(value="q_learning")
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "TCombobox",
        fieldbackground="#313244", background="#313244",
        foreground="#cdd6f4", selectbackground="#45475a",
    )
    agent_box = ttk.Combobox(
        root, textvariable=agent_var,
        values=["q_learning", "sarsa", "dqn"],
        state="readonly", width=18,
        font=("Segoe UI", 10),
    )
    agent_box.pack(padx=20, anchor="w")

    # ── Render ────────────────────────────────────────────────────────────
    section(root, "Render  (pygame window during episodes)")
    render_var = tk.BooleanVar(value=False)
    row2 = tk.Frame(root, bg="#1e1e2e")
    row2.pack(anchor="w", padx=30)
    for label, val in [("ON", True), ("OFF", False)]:
        tk.Radiobutton(
            row2, text=label, variable=render_var, value=val,
            bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
            activebackground="#1e1e2e", font=("Segoe UI", 10),
        ).pack(side="left", padx=6)

    # ── Episodes ──────────────────────────────────────────────────────────
    section(root, "Episodes  (per agent)")
    ep_var = tk.IntVar(value=500)
    tk.Spinbox(
        root, textvariable=ep_var, from_=50, to=5000, increment=50,
        width=8, font=("Segoe UI", 10),
        bg="#313244", fg="#cdd6f4", buttonbackground="#45475a", relief="flat",
    ).pack(padx=20, anchor="w")

    # ── Start / Cancel ────────────────────────────────────────────────────
    def on_start():
        config.mode      = mode_var.get()
        config.agent     = agent_var.get()
        config.render    = render_var.get()
        config.episodes  = int(ep_var.get())
        config.confirmed = True
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_row = tk.Frame(root, bg="#1e1e2e")
    btn_row.pack(pady=20)
    tk.Button(
        btn_row, text="  Start  ", command=on_start,
        font=("Segoe UI", 10, "bold"),
        bg="#a6e3a1", fg="#1e1e2e", relief="flat", padx=8, pady=4,
    ).pack(side="left", padx=10)
    tk.Button(
        btn_row, text=" Cancel ", command=on_cancel,
        font=("Segoe UI", 10),
        bg="#f38ba8", fg="#1e1e2e", relief="flat", padx=8, pady=4,
    ).pack(side="left", padx=10)

    root.mainloop()
    return config


if __name__ == "__main__":
    # quick test: just open the window and print what was chosen
    cfg = show_launcher()
    print(f"mode={cfg.mode!r} agent={cfg.agent!r} "
          f"render={cfg.render} episodes={cfg.episodes} "
          f"confirmed={cfg.confirmed}")
