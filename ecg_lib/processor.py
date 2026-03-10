import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks, butter, filtfilt

class ECGProcessor:
    """Wavelet-based ECG R-peak detector and HR calculator."""

    def __init__(self, time, raw, filtered, fs):
        self.time     = np.asarray(time,     dtype=np.float64)
        self.raw      = np.asarray(raw,      dtype=np.float64)
        self.filtered = np.asarray(filtered, dtype=np.float64)
        self.fs       = float(fs)

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path):
        """Load legacy CSV: columns 'Time (s)', 'Raw_Voltage (V)', 'Filtered_Voltage (V)'."""
        df       = pd.read_csv(path)
        time     = df["Time (s)"].values
        raw      = df["Raw_Voltage (V)"].values
        filtered = df["Filtered_Voltage (V)"].values
        dt       = np.median(np.diff(time))
        fs       = 1.0 / dt if dt > 0 else 125.0
        return cls(time, raw, filtered, fs)

    @classmethod
    def from_vitaldb(cls, path, start_sec=0.0, duration_sec=60.0):
        """Load a VitalDB waveform file (.parquet or .csv).

        Expected columns: Time_sec, SNUADC/ECG_II
        """
        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path, engine="pyarrow")
        else:
            df = pd.read_csv(path)

        end_sec = start_sec + duration_sec
        mask    = (df["Time_sec"] >= start_sec) & (df["Time_sec"] < end_sec)
        df      = df[mask].dropna(subset=["SNUADC/ECG_II"]).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No ECG data in [{start_sec:.0f}s - {end_sec:.0f}s]. "
                "Adjust the start/duration window.")

        time = df["Time_sec"].values
        raw  = df["SNUADC/ECG_II"].values.astype(np.float64)
        dt   = np.median(np.diff(time))
        fs   = 1.0 / dt if dt > 0 else 500.0

        filtered = cls._bandpass(raw, fs, low=0.5, high=40.0)
        return cls(time, raw, filtered, fs)

    @classmethod
    def from_array(cls, time, signal, fs,
                   apply_bandpass=True, bp_low=0.5, bp_high=40.0):
        """Construct directly from NumPy arrays.

        Parameters
        ----------
        time           : array-like - timestamps in seconds
        signal         : array-like - raw ECG amplitude
        fs             : float      - sampling rate (Hz)
        apply_bandpass : bool       - whether to bandpass-filter before storing
        bp_low, bp_high: float      - bandpass cut-offs (Hz)
        """
        raw      = np.asarray(signal, dtype=np.float64)
        filtered = cls._bandpass(raw, fs, bp_low, bp_high) if apply_bandpass else raw.copy()
        return cls(np.asarray(time, dtype=np.float64), raw, filtered, float(fs))

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _bandpass(sig, fs, low=0.5, high=40.0, order=3):
        nyq  = fs / 2.0
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, sig)

    # ── Core algorithms ───────────────────────────────────────────────

    def detect_peaks(self, wavelet="db4", level=7,
                     keep_low=3, keep_high=5,
                     min_bpm=30, max_bpm=200,
                     prominence_frac=0.4, height_frac=0.25):
        """Reconstruct QRS-dominant signal via wavelet decomposition, then detect R-peaks.

        Parameters
        ----------
        wavelet        : str   - PyWavelets wavelet name (default 'db4')
        level          : int   - decomposition depth
        keep_low       : int   - lowest detail level index to retain
        keep_high      : int   - highest detail level index to retain
        min_bpm        : float - minimum physiological heart rate (used for min_distance guard)
        max_bpm        : float - maximum physiological heart rate
        prominence_frac: float - peak prominence as fraction of signal max
        height_frac    : float - peak height threshold as fraction of signal max

        Returns
        -------
        peaks : ndarray - sample indices of detected R-peaks
        qrs   : ndarray - QRS-enhanced signal used for detection
        """
        ecg    = self.filtered.copy()
        coeffs = pywt.wavedec(ecg, wavelet, level=level)

        # Zero out detail levels outside [keep_low, keep_high]
        coeffs_qrs = [
            c if keep_low <= i <= keep_high else np.zeros_like(c)
            for i, c in enumerate(coeffs)
        ]
        qrs = pywt.waverec(coeffs_qrs, wavelet)[: len(ecg)]

        thresh = height_frac    * np.max(np.abs(qrs))
        prom   = prominence_frac * np.max(np.abs(qrs))

        peaks, _ = find_peaks(qrs, height=thresh, prominence=prom)
        return peaks, qrs

    def compute_hr(self, peaks, min_bpm=24, max_bpm=240):
        """Compute RR intervals and heart rate statistics.

        RR intervals outside the physiological window [min_bpm, max_bpm] are
        flagged in ``valid_mask`` and excluded from the mean/std calculations,
        preventing outlier peaks from corrupting the HR estimate.

        Parameters
        ----------
        peaks   : ndarray - sample indices of R-peaks
        min_bpm : float   - lower BPM bound for valid RR intervals (default 24)
        max_bpm : float   - upper BPM bound for valid RR intervals (default 240)

        Returns
        -------
        dict with keys
            rr_intervals : ndarray - all consecutive RR durations (s)
            rr_times     : ndarray - timestamps of the second peak in each pair (s)
            mean_hr      : float   - mean HR computed from *valid* intervals (BPM)
            std_hr       : float   - std of HR over valid intervals (BPM)
            mean_rr      : float   - mean of valid RR intervals (s)
            std_rr       : float   - std of valid RR intervals (s)
            valid_mask   : bool ndarray - True where the RR interval is physiological
        """
        _empty = {
            "rr_intervals": np.array([]),
            "rr_times":     np.array([]),
            "mean_hr":  0.0, "std_hr":  0.0,
            "mean_rr":  0.0, "std_rr":  0.0,
            "valid_mask": np.array([], dtype=bool),
        }
        if len(peaks) < 2:
            return _empty

        rr       = np.diff(peaks) / self.fs        # seconds
        rr_times = self.time[peaks[1:]]

        lo, hi = 60.0 / max_bpm, 60.0 / min_bpm   # convert BPM → seconds
        valid    = (rr >= lo) & (rr <= hi)
        rr_valid = rr[valid]
        hr_vals  = 60.0 / rr_valid if len(rr_valid) else np.array([])

        return {
            "rr_intervals": rr,
            "rr_times":     rr_times,
            "mean_hr":  float(np.mean(hr_vals))   if len(hr_vals)  else 0.0,
            "std_hr":   float(np.std(hr_vals))    if len(hr_vals)  else 0.0,
            "mean_rr":  float(np.mean(rr_valid))  if len(rr_valid) else 0.0,
            "std_rr":   float(np.std(rr_valid))   if len(rr_valid) else 0.0,
            "valid_mask": valid,
        }
