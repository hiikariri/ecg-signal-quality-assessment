import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_lib import ECGProcessor  # noqa: E402

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QGridLayout,
    QSlider, QSpinBox, QDoubleSpinBox, QStatusBar, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ── Matplotlib canvas embedded in Qt ──────────────────────────────────

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, nrows=3, figsize=(12, 8)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.axes = self.fig.subplots(nrows, 1, sharex=(nrows > 1))
        if nrows == 1:
            self.axes = [self.axes]
        super().__init__(self.fig)


# ── Main Window ───────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Peak Detector — R-R Interval & HR Analysis")
        self.setMinimumSize(1200, 800)

        self.processor = None
        self.peaks = None
        self.qrs = None
        self.hr_info = None
        self.current_path = None
        self.is_vitaldb = False

        self._build_ui()
        self.statusBar().showMessage("Load a CSV or VitalDB Parquet file to begin.")

    # ── UI construction ───────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left panel: controls + stats
        left = QVBoxLayout()
        left.addWidget(self._make_file_group())
        left.addWidget(self._make_param_group())
        left.addWidget(self._make_stats_group())
        left.addWidget(self._make_rr_table())
        left.addStretch()

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(320)

        # Right panel: plots
        self.canvas = MplCanvas(nrows=3)
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

        self.btn_load = QPushButton("Load ECG File …")
        self.btn_load.clicked.connect(self._on_load)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        lay.addWidget(self.btn_load)
        lay.addWidget(self.lbl_file)

        # VitalDB segment window controls
        self.grp_segment = QGroupBox("VitalDB Segment")
        seg_grid = QGridLayout()
        seg_grid.addWidget(QLabel("Start (s):"), 0, 0)
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, 999999)
        self.spin_start.setValue(0)
        seg_grid.addWidget(self.spin_start, 0, 1)

        seg_grid.addWidget(QLabel("Duration (s):"), 1, 0)
        self.spin_dur = QSpinBox()
        self.spin_dur.setRange(10, 600)
        self.spin_dur.setValue(60)
        self.spin_dur.setSuffix(" s")
        seg_grid.addWidget(self.spin_dur, 1, 1)

        self.btn_reload = QPushButton("↺  Reload Segment")
        self.btn_reload.setEnabled(False)
        self.btn_reload.clicked.connect(self._on_reload_segment)
        seg_grid.addWidget(self.btn_reload, 2, 0, 1, 2)
        self.grp_segment.setLayout(seg_grid)
        self.grp_segment.setVisible(False)
        lay.addWidget(self.grp_segment)

        grp.setLayout(lay)
        return grp

    def _make_param_group(self):
        grp = QGroupBox("Detection Parameters")
        grid = QGridLayout()

        # Signal selector
        grid.addWidget(QLabel("Signal:"), 0, 0)
        self.cmb_signal = QComboBox()
        self.cmb_signal.addItems(["Filtered", "Raw"])
        grid.addWidget(self.cmb_signal, 0, 1)

        # Wavelet level
        grid.addWidget(QLabel("Wavelet level:"), 1, 0)
        self.spin_level = QSpinBox()
        self.spin_level.setRange(3, 10)
        self.spin_level.setValue(7)
        grid.addWidget(self.spin_level, 1, 1)

        # Keep low
        grid.addWidget(QLabel("Keep coeff low:"), 2, 0)
        self.spin_kl = QSpinBox()
        self.spin_kl.setRange(0, 10)
        self.spin_kl.setValue(3)
        grid.addWidget(self.spin_kl, 2, 1)

        # Keep high
        grid.addWidget(QLabel("Keep coeff high:"), 3, 0)
        self.spin_kh = QSpinBox()
        self.spin_kh.setRange(0, 10)
        self.spin_kh.setValue(5)
        grid.addWidget(self.spin_kh, 3, 1)

        # Prominence fraction
        grid.addWidget(QLabel("Prominence (%):"), 4, 0)
        self.spin_prom = QSpinBox()
        self.spin_prom.setRange(5, 90)
        self.spin_prom.setValue(40)
        self.spin_prom.setSuffix(" %")
        grid.addWidget(self.spin_prom, 4, 1)

        # Height fraction
        grid.addWidget(QLabel("Height thresh (%):"), 5, 0)
        self.spin_ht = QSpinBox()
        self.spin_ht.setRange(5, 90)
        self.spin_ht.setValue(25)
        self.spin_ht.setSuffix(" %")
        grid.addWidget(self.spin_ht, 5, 1)

        self.btn_detect = QPushButton("▶  Detect Peaks")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._on_detect)
        grid.addWidget(self.btn_detect, 6, 0, 1, 2)

        grp.setLayout(grid)
        return grp

    def _make_stats_group(self):
        grp = QGroupBox("Results")
        grid = QGridLayout()
        bold = QFont()
        bold.setBold(True)

        labels = [
            ("Sampling Rate:", "fs"), ("Peaks Found:", "npeaks"),
            ("Mean HR:", "mean_hr"), ("Std HR:", "std_hr"),
            ("Mean RR:", "mean_rr"), ("Std RR:", "std_rr"),
        ]
        self.stat_labels = {}
        for row, (text, key) in enumerate(labels):
            lbl = QLabel(text)
            val = QLabel("—")
            val.setFont(bold)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(val, row, 1)
            self.stat_labels[key] = val

        grp.setLayout(grid)
        return grp

    def _make_rr_table(self):
        grp = QGroupBox("R-R Intervals")
        lay = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Beat #", "RR (s)", "HR (BPM)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMaximumHeight(220)
        lay.addWidget(self.table)
        grp.setLayout(lay)
        return grp

    # ── Slots ─────────────────────────────────────────────────────────
    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ECG File", "",
            "All Supported (*.parquet *.csv);;"
            "VitalDB Parquet (*.parquet);;"
            "CSV Files (*.csv);;"
            "All Files (*)")
        if not path:
            return
        self.current_path = path
        self.is_vitaldb = path.lower().endswith('.parquet') or self._is_vitaldb_csv(path)
        self._load_from_path()

    def _is_vitaldb_csv(self, path):
        """Peek at the CSV header to check if it is a VitalDB-format file."""
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            return 'SNUADC/ECG_II' in header
        except Exception:
            return False

    def _load_from_path(self):
        """(Re-)load data from self.current_path using current segment settings."""
        try:
            if self.is_vitaldb:
                self.processor = ECGProcessor.from_vitaldb(
                    self.current_path,
                    start_sec=float(self.spin_start.value()),
                    duration_sec=float(self.spin_dur.value()),
                )
                self.grp_segment.setVisible(True)
                self.btn_reload.setEnabled(True)
            else:
                self.processor = ECGProcessor.from_csv(self.current_path)
                self.grp_segment.setVisible(False)

            self.lbl_file.setText(os.path.basename(self.current_path))
            self.stat_labels['fs'].setText(f"{self.processor.fs:.1f} Hz")
            self.btn_detect.setEnabled(True)
            self.statusBar().showMessage(
                f"Loaded {len(self.processor.time)} samples  "
                f"({self.processor.time[0]:.1f}s – {self.processor.time[-1]:.1f}s)")
            self._on_detect()
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {e}")

    def _on_reload_segment(self):
        if self.current_path:
            self._load_from_path()

    def _on_detect(self):
        if self.processor is None:
            return

        # Swap signal if user chose Raw
        if self.cmb_signal.currentText() == "Raw":
            self.processor.filtered = self.processor.raw.copy()

        self.peaks, self.qrs = self.processor.detect_peaks(
            level=self.spin_level.value(),
            keep_low=self.spin_kl.value(),
            keep_high=self.spin_kh.value(),
            prominence_frac=self.spin_prom.value() / 100.0,
            height_frac=self.spin_ht.value() / 100.0,
        )
        self.hr_info = self.processor.compute_hr(self.peaks)
        self._update_stats()
        self._update_table()
        self._draw()
        self.statusBar().showMessage(
            f"Detected {len(self.peaks)} R-peaks  |  "
            f"HR = {self.hr_info['mean_hr']:.1f} ± {self.hr_info['std_hr']:.1f} BPM")

    # ── Display helpers ───────────────────────────────────────────────
    def _update_stats(self):
        info = self.hr_info
        self.stat_labels['npeaks'].setText(str(len(self.peaks)))
        self.stat_labels['mean_hr'].setText(f"{info['mean_hr']:.1f} BPM")
        self.stat_labels['std_hr'].setText(f"{info['std_hr']:.2f} BPM")
        self.stat_labels['mean_rr'].setText(f"{info['mean_rr']*1000:.1f} ms")
        self.stat_labels['std_rr'].setText(f"{info['std_rr']*1000:.2f} ms")

    def _update_table(self):
        rr = self.hr_info['rr_intervals']
        self.table.setRowCount(len(rr))
        for i, interval in enumerate(rr):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{interval:.4f}"))
            hr = 60.0 / interval if interval > 0 else 0
            self.table.setItem(i, 2, QTableWidgetItem(f"{hr:.1f}"))

    def _draw(self):
        p = self.processor
        t, ecg = p.time, p.filtered
        peaks, qrs = self.peaks, self.qrs
        info = self.hr_info

        for ax in self.canvas.axes:
            ax.clear()

        # ── Ax 0: ECG + detected R-peaks ──
        ax0 = self.canvas.axes[0]
        ax0.plot(t, ecg, color='#1f77b4', lw=0.8, label='ECG Signal')
        ax0.scatter(t[peaks], ecg[peaks], c='red', s=40, zorder=5,
                    label=f'R-peaks ({len(peaks)})')
        # vertical lines at each peak
        for pk in peaks:
            ax0.axvline(t[pk], color='red', alpha=0.15, lw=0.5)
        ax0.set_ylabel('Voltage (V)')
        ax0.set_title('ECG Signal with Detected R-Peaks', fontweight='bold')
        ax0.legend(loc='upper right', fontsize=8)
        ax0.grid(True, alpha=0.3)

        # ── Ax 1: QRS-enhanced wavelet reconstruction ──
        ax1 = self.canvas.axes[1]
        ax1.plot(t, qrs, color='#2ca02c', lw=0.8, label='QRS-enhanced (wavelet)')
        ax1.scatter(t[peaks], qrs[peaks], c='red', s=30, zorder=5)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Wavelet QRS Reconstruction (db4)', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Ax 2: R-R intervals tachogram ──
        ax2 = self.canvas.axes[2]
        if len(info['rr_intervals']):
            rr_ms = info['rr_intervals'] * 1000
            colors = ['green' if v else 'red' for v in info['valid_mask']]
            ax2.bar(info['rr_times'], rr_ms, width=0.3, color=colors, alpha=0.7)
            ax2.axhline(info['mean_rr'] * 1000, color='blue', ls='--', lw=1,
                        label=f"Mean RR = {info['mean_rr']*1000:.0f} ms")
            ax2.set_ylabel('R-R interval (ms)')
            ax2.legend(loc='upper right', fontsize=8)
        ax2.set_xlabel('Time (s)')
        ax2.set_title('R-R Interval Tachogram', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        self.canvas.draw()


# ── Entry point ───────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()

    # Auto-load VitalDB parquet if present, else fall back to legacy CSV
    vitaldb_default = os.path.join(
        os.path.dirname(__file__), '..', 'bgl_estimation_model',
        'vitaldb_data', 'waveforms', 'case_1_waves.parquet')
    legacy_csv = os.path.join(os.path.dirname(__file__), '..', 'framework', 'ecgload.csv')

    if os.path.isfile(vitaldb_default):
        win.current_path = vitaldb_default
        win.is_vitaldb = True
        win._load_from_path()
    elif os.path.isfile(legacy_csv):
        win.current_path = legacy_csv
        win.is_vitaldb = False
        win._load_from_path()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
