import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_lib import ECGProcessor, ECGSQAEngine  # noqa: E402

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QSplitter, QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ══════════════════════════════════════════════════════════════════════
#  Matplotlib canvas
# ══════════════════════════════════════════════════════════════════════

class SQACanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig  = Figure(figsize=(12, 9), tight_layout=True)
        self.axes = self.fig.subplots(3, 1, sharex=False)
        super().__init__(self.fig)


# ══════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════

class SQAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Signal Quality Assessment (SQA)")
        self.setMinimumSize(1300, 860)

        self.processor    = None
        self.sqa_result   = None
        self.current_path = None
        self.is_vitaldb   = False

        self._build_ui()
        self.statusBar().showMessage("Load a CSV or VitalDB Parquet file to begin.")

    # ── UI construction ───────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        left.addWidget(self._make_file_group())
        left.addWidget(self._make_threshold_group())
        left.addWidget(self._make_result_panel())
        left.addWidget(self._make_stages_group())
        left.addStretch()

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(300)

        self.canvas  = SQACanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)

        right = QVBoxLayout()
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas)
        right_w = QWidget()
        right_w.setLayout(right)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _make_file_group(self):
        grp = QGroupBox("Data Source")
        lay = QVBoxLayout()

        self.btn_load = QPushButton("Load ECG File \u2026")
        self.btn_load.clicked.connect(self._on_load)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        lay.addWidget(self.btn_load)
        lay.addWidget(self.lbl_file)

        # VitalDB segment controls
        self.grp_seg = QGroupBox("VitalDB Segment")
        sg = QGridLayout()
        sg.addWidget(QLabel("Start (s):"), 0, 0)
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, 999999)
        sg.addWidget(self.spin_start, 0, 1)
        sg.addWidget(QLabel("Duration (s):"), 1, 0)
        self.spin_dur = QSpinBox()
        self.spin_dur.setRange(10, 600)
        self.spin_dur.setValue(60)
        self.spin_dur.setSuffix(" s")
        sg.addWidget(self.spin_dur, 1, 1)
        self.btn_reload = QPushButton("\u21ba  Reload Segment")
        self.btn_reload.setEnabled(False)
        self.btn_reload.clicked.connect(self._on_reload)
        sg.addWidget(self.btn_reload, 2, 0, 1, 2)
        self.grp_seg.setLayout(sg)
        self.grp_seg.setVisible(False)
        lay.addWidget(self.grp_seg)

        grp.setLayout(lay)
        return grp

    def _make_threshold_group(self):
        grp = QGroupBox("SQA Thresholds")
        grid = QGridLayout()

        grid.addWidget(QLabel("Flat thresh (%):"), 0, 0)
        self.spin_flat = QSpinBox()
        self.spin_flat.setRange(10, 99)
        self.spin_flat.setValue(80)
        self.spin_flat.setSuffix(" %")
        grid.addWidget(self.spin_flat, 0, 1)

        grid.addWidget(QLabel("Noise NZC thresh:"), 1, 0)
        self.spin_nzc = QSpinBox()
        self.spin_nzc.setRange(50, 2000)
        self.spin_nzc.setValue(200)
        grid.addWidget(self.spin_nzc, 1, 1)

        grid.addWidget(QLabel("HR min (BPM):"), 2, 0)
        self.spin_hrmin = QSpinBox()
        self.spin_hrmin.setRange(5, 50)
        self.spin_hrmin.setValue(24)
        grid.addWidget(self.spin_hrmin, 2, 1)

        grid.addWidget(QLabel("HR max (BPM):"), 3, 0)
        self.spin_hrmax = QSpinBox()
        self.spin_hrmax.setRange(100, 350)
        self.spin_hrmax.setValue(240)
        grid.addWidget(self.spin_hrmax, 3, 1)

        grid.addWidget(QLabel("QRS amp CoV:"), 4, 0)
        self.dspin_cov = QDoubleSpinBox()
        self.dspin_cov.setRange(0.05, 2.00)
        self.dspin_cov.setSingleStep(0.05)
        self.dspin_cov.setDecimals(2)
        self.dspin_cov.setValue(0.90)
        grid.addWidget(self.dspin_cov, 4, 1)

        self.btn_assess = QPushButton("\u25b6  Run SQA")
        self.btn_assess.setEnabled(False)
        self.btn_assess.clicked.connect(self._on_assess)
        grid.addWidget(self.btn_assess, 5, 0, 1, 2)

        grp.setLayout(grid)
        return grp

    def _make_result_panel(self):
        grp = QGroupBox("Assessment Result")
        lay = QVBoxLayout()

        self.lbl_result = QLabel("\u2014")
        f = QFont()
        f.setPointSize(11)
        f.setBold(True)
        self.lbl_result.setFont(f)
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setMinimumHeight(56)
        self.lbl_result.setFrameShape(QFrame.StyledPanel)
        lay.addWidget(self.lbl_result)

        grp.setLayout(lay)
        return grp

    def _make_stages_group(self):
        grp = QGroupBox("Stage Details")
        grid = QGridLayout()

        bold = QFont()
        bold.setBold(True)

        rows = [
            ("Stage 1 \u2013 Flat/Sat:", "s1"),
            ("  Flat proportion:",       "flat_prop"),
            ("Stage 2 \u2013 Noise:",    "s2"),
            ("  Zero crossings:",        "nzc"),
            ("Stage 3 \u2013 QRS/HR:",   "s3"),
            ("  Mean HR:",               "mean_hr"),
            ("  QRS amp CoV:",           "amp_cov"),
        ]

        self.slbl = {}
        for row, (text, key) in enumerate(rows):
            lbl = QLabel(text)
            val = QLabel("\u2014")
            val.setFont(bold)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(val, row, 1)
            self.slbl[key] = val

        grp.setLayout(grid)
        return grp

    # ── Slots ─────────────────────────────────────────────────────────
    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ECG File", "",
            "Supported (*.parquet *.csv);;All Files (*)")
        if not path:
            return
        self.current_path = path
        self.is_vitaldb = (
            path.lower().endswith(".parquet") or self._peek_vitaldb(path))
        self._load()

    def _peek_vitaldb(self, path):
        try:
            return "SNUADC/ECG_II" in pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            return False

    def _load(self):
        try:
            if self.is_vitaldb:
                self.processor = ECGProcessor.from_vitaldb(
                    self.current_path,
                    start_sec=float(self.spin_start.value()),
                    duration_sec=float(self.spin_dur.value()),
                )
                self.grp_seg.setVisible(True)
                self.btn_reload.setEnabled(True)
            else:
                self.processor = ECGProcessor.from_csv(self.current_path)
                self.grp_seg.setVisible(False)

            self.lbl_file.setText(os.path.basename(self.current_path))
            self.btn_assess.setEnabled(True)
            n  = len(self.processor.time)
            t0 = self.processor.time[0]
            t1 = self.processor.time[-1]
            self.statusBar().showMessage(
                f"Loaded {n} samples  ({t0:.1f}s \u2013 {t1:.1f}s)  "
                f"fs = {self.processor.fs:.0f} Hz")
            self._on_assess()
        except Exception as e:
            self.statusBar().showMessage(f"Load error: {e}")

    def _on_reload(self):
        if self.current_path:
            self._load()

    def _on_assess(self):
        if self.processor is None:
            return

        engine = ECGSQAEngine(self.processor)
        engine.FLAT_THRESH    = self.spin_flat.value() / 100.0
        engine.NZC_THRESH     = self.spin_nzc.value()
        engine.HR_MIN         = self.spin_hrmin.value()
        engine.HR_MAX         = self.spin_hrmax.value()
        engine.QRS_COV_THRESH = self.dspin_cov.value()

        self.sqa_result = engine.assess()
        self._update_panels()
        self._draw()
        self.statusBar().showMessage(
            "SQA: " + self.sqa_result["label"].replace("\n", " "))

    # ── Panel updates ─────────────────────────────────────────────────
    def _update_panels(self):
        r = self.sqa_result

        disp = r["label"].replace("\n", " ")
        if r["quality"] == "acceptable":
            style = ("background:#d4edda; color:#155724; "
                     "border-radius:6px; padding:6px;")
        else:
            style = ("background:#f8d7da; color:#721c24; "
                     "border-radius:6px; padding:6px;")
        self.lbl_result.setStyleSheet(style)
        self.lbl_result.setText(disp)

        def tick(ok):
            return "\u2714 PASS" if ok else "\u2718 FAIL"

        sl = self.slbl

        # Stage 1
        sl["s1"].setText(tick(not r["is_flat"]))
        p = r["flat_proportion"]
        sl["flat_prop"].setText(f"{p*100:.1f} %" if p is not None else "\u2014")

        # Stage 2
        if r["nzc"] is not None:
            sl["s2"].setText(tick(not r["is_noise"]))
            sl["nzc"].setText(str(r["nzc"]))
        else:
            sl["s2"].setText("\u2014 (not run)")
            sl["nzc"].setText("\u2014")

        # Stage 3
        if r["stage_reached"] >= 3:
            hr_ok  = (self.spin_hrmin.value()
                      <= (r["mean_hr"] or 0.0)
                      <= self.spin_hrmax.value())
            cov_ok = (r["amp_cov"] is not None
                      and r["amp_cov"] < self.dspin_cov.value())
            sl["s3"].setText(tick(hr_ok and cov_ok))
            sl["mean_hr"].setText(
                f"{r['mean_hr']:.1f} BPM" if r["mean_hr"] is not None else "\u2014")
            sl["amp_cov"].setText(
                f"{r['amp_cov']:.3f}" if r["amp_cov"] is not None else "\u2014")
        else:
            sl["s3"].setText("\u2014 (not run)")
            sl["mean_hr"].setText("\u2014")
            sl["amp_cov"].setText("\u2014")

    # ── Drawing ───────────────────────────────────────────────────────
    def _draw(self):
        for ax in self.canvas.axes:
            ax.cla()

        proc = self.processor
        res  = self.sqa_result
        t    = proc.time
        sig  = proc.filtered

        RED   = "red"
        BLACK = "black"
        GRAY  = "gray"

        # ── Ax 0: ECG signal + flat-region highlights + R-peaks ──────
        ax0 = self.canvas.axes[0]
        ax0.plot(t, sig, color="#1f77b4", lw=0.7, label="ECG (filtered)")

        if res["is_flat"] and res["absdiff_norm"] is not None:
            flat_ref = np.concatenate([[False], res["absdiff_norm"] <= 1.0])
            first = True
            for s, e in _find_spans(flat_ref):
                kw = ({"color": "red", "alpha": 0.25,
                        "label": "Flat/saturated region"}
                      if first else {"color": "red", "alpha": 0.25})
                ax0.axvspan(t[s], t[min(e, len(t) - 1)], **kw)
                first = False

        if res["peaks"] is not None and len(res["peaks"]) > 0:
            pk = res["peaks"]
            ax0.scatter(t[pk], sig[pk], c="red", s=25, zorder=5,
                        label=f"R-peaks ({len(pk)})")

        fp     = res["flat_proportion"]
        fp_str = f"{fp*100:.1f} %" if fp is not None else "N/A"
        ax0.set_title(f"Stage 1 \u2013 Flat / Saturation   "
                      f"[flat proportion = {fp_str}]",
                      color=RED if res["is_flat"] else BLACK,
                      fontweight="bold")
        ax0.set_ylabel("Voltage (V)")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True, alpha=0.3)

        # ── Ax 1: Zero-crossing view (Stage 2) ────────────────────────
        ax1 = self.canvas.axes[1]
        if res["sig_sc"] is not None:
            sig_sc = res["sig_sc"]
            ax1.plot(t, sig_sc, color="#ff7f0e", lw=0.7,
                     label="ECG scaled [\u22121, 1]")
            ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)

            signs = np.sign(sig_sc)
            nz_i  = np.where(signs != 0)[0]
            if len(nz_i) > 1:
                cx_mask = np.diff(signs[nz_i]) != 0
                cx_idx  = nz_i[1:][cx_mask]
                ax1.scatter(t[cx_idx], np.zeros(len(cx_idx)),
                            c="red", s=6, zorder=5,
                            label=f"Zero crossings ({res['nzc']})")

            nzc_val = res["nzc"] if res["nzc"] is not None else "N/A"
            ax1.set_title(
                f"Stage 2 \u2013 Pure Noise   "
                f"[NZC = {nzc_val},  threshold = {self.spin_nzc.value()}]",
                color=RED if res["is_noise"] else BLACK,
                fontweight="bold")
            ax1.set_ylabel("Amplitude")
            ax1.legend(loc="upper right", fontsize=8)
        else:
            ax1.set_title("Stage 2 \u2013 Pure Noise   [not reached]",
                          color=GRAY, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # ── Ax 2: RR tachogram — Stage 3 ─────────────────────────────
        ax2 = self.canvas.axes[2]
        hr_info = res.get("hr_info")
        if hr_info and len(hr_info.get("rr_intervals", [])) > 0:
            rr_ms  = hr_info["rr_intervals"] * 1000.0
            rr_t   = hr_info["rr_times"]
            valid  = hr_info.get("valid_mask",
                                 np.ones(len(rr_ms), dtype=bool))
            colors = ["#2ca02c" if v else "#d62728" for v in valid]
            ax2.bar(rr_t, rr_ms, width=0.4, color=colors, alpha=0.75)

            if hr_info["mean_rr"]:
                ax2.axhline(hr_info["mean_rr"] * 1000, color="blue",
                            ls="--", lw=1,
                            label=f"Mean RR = {hr_info['mean_rr']*1000:.0f} ms")

            hr_v    = res["mean_hr"]
            cov_v   = res["amp_cov"]
            hr_str  = f"{hr_v:.1f} BPM" if hr_v  is not None else "N/A"
            cov_str = (f"   |   QRS CoV = {cov_v:.3f}"
                       if cov_v is not None else "")

            stage3_fail = (res["stage_reached"] >= 3
                           and res["quality"] == "unacceptable")
            ax2.set_title(
                f"Stage 3 \u2013 QRS / HR   [HR = {hr_str}{cov_str}]",
                color=RED if stage3_fail else BLACK,
                fontweight="bold")
            ax2.set_ylabel("R-R interval (ms)")
            ax2.legend(loc="upper right", fontsize=8)
        else:
            stage_msg = ("not reached"
                         if res["stage_reached"] < 3
                         else "no peaks detected")
            ax2.set_title(
                f"Stage 3 \u2013 QRS / HR   [{stage_msg}]",
                color=GRAY if res["stage_reached"] < 3 else RED,
                fontweight="bold")

        ax2.set_xlabel("Time (s)")
        ax2.grid(True, alpha=0.3)

        self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════

def _find_spans(mask):
    """Return list of (start, end) index pairs for contiguous True runs."""
    spans  = []
    in_run = False
    start  = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start  = i
            in_run = True
        elif not v and in_run:
            spans.append((start, i - 1))
            in_run = False
    if in_run:
        spans.append((start, len(mask) - 1))
    return spans


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SQAWindow()
    win.show()

    # Auto-load defaults (same precedence as ecg_peak_detector.py)
    vitaldb_default = os.path.join(
        os.path.dirname(__file__), "..", "bgl_estimation_model",
        "vitaldb_data", "waveforms", "case_1_waves.parquet")
    legacy_csv = os.path.join(
        os.path.dirname(__file__), "..", "framework", "ecgload.csv")

    if os.path.isfile(vitaldb_default):
        win.current_path = vitaldb_default
        win.is_vitaldb   = True
        win._load()
    elif os.path.isfile(legacy_csv):
        win.current_path = legacy_csv
        win.is_vitaldb   = False
        win._load()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
