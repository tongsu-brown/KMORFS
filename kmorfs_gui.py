"""
KMORFS GUI Launcher
Four-mode interface for the KMORFS stress fitting suite.

Author : Tong Su
Lab    : Chason Research Group, Brown University
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON = r"D:\anaconda3\envs\data2060new\python.exe"
BASE   = Path(__file__).resolve().parent

MODES = [
    {
        "id":    1,
        "title": "General Stress-Thickness",
        "desc":  (
            "Fits stress-thickness data using shared material\n"
            "parameters across multiple datasets. Supports\n"
            "multi-material joint fitting via database CSVs."
        ),
        "script": BASE / "general_stress_thickness"          / "fit_general_stress.py",
        "cwd":    BASE / "general_stress_thickness",
        "color":  "#2563eb",   # blue
    },
    {
        "id":    2,
        "title": "Incremental (Steady-State) Stress",
        "desc":  (
            "Fits steady-state stress-free (SSSF) incremental\n"
            "stress data. Uses numpy batch stress equation\n"
            "with scipy L-BFGS-B optimization."
        ),
        "script": BASE / "incremental_stress"                / "fit_incremental_stress.py",
        "cwd":    BASE / "incremental_stress",
        "color":  "#16a34a",   # green
    },
    {
        "id":    3,
        "title": "Alloy Extension",
        "desc":  (
            "Alloy stress-thickness fitting. Pure element\n"
            "parameters are blended via rule-of-mixtures.\n"
            "List pure elements first in MATERIALS."
        ),
        "script": BASE / "alloy_extension_stress_thickness"  / "fit_alloy_stress.py",
        "cwd":    BASE / "alloy_extension_stress_thickness",
        "color":  "#d97706",   # amber
    },
    {
        "id":    4,
        "title": "Early-State Nucleation",
        "desc":  (
            "Fits the nucleation / coalescence regime using\n"
            "ellipsoidal grain-cap geometry and grain\n"
            "boundary area derivatives (dA/dt)."
        ),
        "script": BASE / "early_state_stress_thickness"      / "fit_early_state_stress.py",
        "cwd":    BASE / "early_state_stress_thickness",
        "color":  "#7c3aed",   # violet
    },
]

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#f3f4f6"
CARD_BG   = "#ffffff"
TITLE_FG  = "#111827"
MUTED_FG  = "#6b7280"
LOG_BG    = "#1e1e2e"
LOG_FG    = "#cdd6f4"
SBAR_BG   = "#e5e7eb"


# ── Helper: rounded button (plain tk.Button with flat relief) ─────────────────
def _make_btn(parent, text, color, command, state="normal"):
    return tk.Button(
        parent,
        text=text,
        font=("Times New Roman", 10, "bold"),
        bg=color, fg="white",
        activebackground=color, activeforeground="white",
        relief="flat", bd=0,
        padx=12, pady=4,
        cursor="hand2",
        state=state,
        command=command,
    )


# ── Application ───────────────────────────────────────────────────────────────
class KMORFSApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("KMORFS")
        self.configure(bg=BG)
        self.geometry("900x740")
        self.minsize(820, 660)

        # running subprocesses keyed by mode id
        self._procs: dict[int, subprocess.Popen] = {}

        self._build_header()
        self._build_grid()
        self._build_log_section()
        self._build_statusbar()

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        f = tk.Frame(self, bg=BG)
        f.pack(fill="x", padx=32, pady=(18, 6))

        tk.Label(
            f,
            text="KMORFS  —  Kinetic Model of Residual Film Stress",
            font=("Times New Roman", 16, "bold"),
            bg=BG, fg=TITLE_FG,
        ).pack(anchor="center")

        tk.Label(
            f,
            text="Produced by Chason Research Group, Brown University",
            font=("Times New Roman", 11),
            bg=BG, fg=MUTED_FG,
        ).pack(anchor="center", pady=(2, 10))

        ttk.Separator(f, orient="horizontal").pack(fill="x")

    # ── Mode grid (2 × 2) ────────────────────────────────────────────────────
    def _build_grid(self):
        gf = tk.Frame(self, bg=BG)
        gf.pack(fill="x", padx=28, pady=(12, 4))
        gf.columnconfigure(0, weight=1, uniform="col")
        gf.columnconfigure(1, weight=1, uniform="col")

        for i, mode in enumerate(MODES):
            row, col = divmod(i, 2)
            card = self._build_card(gf, mode)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

    def _build_card(self, parent, mode: dict) -> tk.Frame:
        color = mode["color"]

        # Outer wrapper — gives the coloured 3-px top edge look
        outer = tk.Frame(parent, bg=color, bd=0, highlightthickness=0)

        # Top badge strip
        strip = tk.Frame(outer, bg=color, height=40)
        strip.pack(fill="x")
        strip.pack_propagate(False)

        tk.Label(
            strip,
            text=f"  Mode {mode['id']}",
            font=("Times New Roman", 12, "bold"),
            bg=color, fg="white", anchor="w",
        ).pack(side="left", padx=10, fill="y")

        # White body
        body = tk.Frame(outer, bg=CARD_BG, padx=14, pady=10)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text=mode["title"],
            font=("Times New Roman", 11, "bold"),
            bg=CARD_BG, fg=TITLE_FG, anchor="w",
        ).pack(anchor="w")

        tk.Label(
            body,
            text=mode["desc"],
            font=("Times New Roman", 10),
            bg=CARD_BG, fg=MUTED_FG,
            justify="left", anchor="w",
        ).pack(anchor="w", pady=(4, 8))

        # Button row
        btn_row = tk.Frame(body, bg=CARD_BG)
        btn_row.pack(fill="x")

        run_btn  = _make_btn(btn_row, "  Run  ", color,   lambda m=mode: self._run(m))
        stop_btn = _make_btn(btn_row, " Stop ", "#dc2626", lambda m=mode: self._stop(m),
                             state="disabled")
        open_btn = _make_btn(btn_row, "Folder", "#6b7280", lambda m=mode: self._open_folder(m))

        run_btn .pack(side="left")
        stop_btn.pack(side="left", padx=(6, 0))
        open_btn.pack(side="left", padx=(6, 0))

        status_lbl = tk.Label(
            btn_row,
            text="Idle",
            font=("Times New Roman", 10),
            bg=CARD_BG, fg="#9ca3af",
        )
        status_lbl.pack(side="right", padx=4)

        # Save refs for runtime updates
        mode["_run"]    = run_btn
        mode["_stop"]   = stop_btn
        mode["_status"] = status_lbl

        return outer

    # ── Log section ──────────────────────────────────────────────────────────
    def _build_log_section(self):
        lf = tk.Frame(self, bg=BG)
        lf.pack(fill="both", expand=True, padx=28, pady=(4, 2))

        hdr = tk.Frame(lf, bg=BG)
        hdr.pack(fill="x", pady=(0, 4))

        tk.Label(
            hdr,
            text="Output Log",
            font=("Times New Roman", 11, "bold"),
            bg=BG, fg=TITLE_FG,
        ).pack(side="left")

        tk.Button(
            hdr, text="Clear",
            font=("Times New Roman", 10),
            bg=SBAR_BG, fg="#374151",
            relief="flat", cursor="hand2",
            padx=8, pady=2,
            command=self._clear_log,
        ).pack(side="right")

        self._log = scrolledtext.ScrolledText(
            lf,
            font=("Consolas", 9),
            bg=LOG_BG, fg=LOG_FG,
            insertbackground=LOG_FG,
            relief="flat",
            height=10,
            state="disabled",
        )
        self._log.pack(fill="both", expand=True)

    # ── Status bar ───────────────────────────────────────────────────────────
    def _build_statusbar(self):
        sb = tk.Frame(self, bg=SBAR_BG, height=26)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(
            sb,
            textvariable=self._status_var,
            font=("Times New Roman", 10),
            bg=SBAR_BG, fg="#374151",
            anchor="w",
        ).pack(side="left", padx=12, fill="y")

        tk.Label(
            sb,
            text="Copyright \u00a9 2025  Chason Research Group, Brown University",
            font=("Times New Roman", 9),
            bg=SBAR_BG, fg="#9ca3af",
        ).pack(side="right", padx=12, fill="y")

    # ── Actions ───────────────────────────────────────────────────────────────
    def _run(self, mode: dict):
        mid = mode["id"]
        if mid in self._procs:
            self._log_write(f"[Mode {mid}] Already running — please wait or stop it first.\n")
            return

        script = mode["script"]
        if not script.exists():
            self._log_write(
                f"[Mode {mid}] ERROR: script not found:\n  {script}\n"
            )
            return

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
        env["PYTHONIOENCODING"]       = "utf-8"
        env["PYTHONUNBUFFERED"]       = "1"
        # Do NOT force Agg — let scripts display their own plot windows

        proc = subprocess.Popen(
            [PYTHON, "-u", str(script)],
            cwd=str(mode["cwd"]),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._procs[mid] = proc

        mode["_run"]   .configure(state="disabled")
        mode["_stop"]  .configure(state="normal")
        mode["_status"].configure(text="Running...", fg="#d97706")
        self._status_var.set(f"Mode {mid} — {mode['title']} — running...")
        self._log_write(
            f"\n{'─'*60}\n"
            f"[Mode {mid}]  {mode['title']}\n"
            f"Script : {script}\n"
            f"{'─'*60}\n"
        )

        threading.Thread(
            target=self._stream, args=(mode,), daemon=True
        ).start()

    def _stream(self, mode: dict):
        proc = self._procs.get(mode["id"])
        if proc is None:
            return
        for line in proc.stdout:
            self._log_write(line)
        proc.wait()
        self.after(0, self._finished, mode, proc.returncode)

    def _finished(self, mode: dict, rc: int):
        mid = mode["id"]
        self._procs.pop(mid, None)
        mode["_run"] .configure(state="normal")
        mode["_stop"].configure(state="disabled")
        if rc == 0:
            mode["_status"].configure(text="Done \u2713", fg="#16a34a")
            self._log_write(f"[Mode {mid}] Finished successfully.\n")
        else:
            mode["_status"].configure(text=f"Error (rc={rc})", fg="#dc2626")
            self._log_write(f"[Mode {mid}] Exited with code {rc}.\n")
        self._status_var.set("Ready")

    def _stop(self, mode: dict):
        proc = self._procs.get(mode["id"])
        if proc:
            proc.terminate()
            self._log_write(f"[Mode {mode['id']}] Terminated by user.\n")

    def _open_folder(self, mode: dict):
        folder = mode["cwd"]
        if folder.exists():
            os.startfile(str(folder))          # Windows Explorer
        else:
            self._log_write(f"[Mode {mode['id']}] Folder not found: {folder}\n")

    # ── Thread-safe log write ─────────────────────────────────────────────────
    def _log_write(self, text: str):
        def _do():
            self._log.configure(state="normal")
            self._log.insert("end", text)
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _do)

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = KMORFSApp()
    app.mainloop()
