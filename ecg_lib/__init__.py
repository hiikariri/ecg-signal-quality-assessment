"""ecg_lib – ECG processing and signal quality assessment library
================================================================
Pure-Python, GUI-free package providing:

    ECGProcessor  – wavelet-based R-peak detection and HR calculation
    ECGSQAEngine  – three-stage ECG signal quality assessment pipeline

Quick start
-----------
    from ecg_lib import ECGProcessor, ECGSQAEngine

    # Load data
    proc = ECGProcessor.from_csv("ecgload.csv")
    # or
    proc = ECGProcessor.from_vitaldb("case_1_waves.parquet", start_sec=0, duration_sec=60)
    # or
    proc = ECGProcessor.from_array(time_array, ecg_array, fs=500)

    # Peak detection
    peaks, qrs_signal = proc.detect_peaks()
    hr_info = proc.compute_hr(peaks)           # uses default BPM limits [24, 240]
    print(f"HR = {hr_info['mean_hr']:.1f} BPM")

    # Signal quality assessment
    engine = ECGSQAEngine(proc)
    result = engine.assess()
    print(result["quality"], result["label"])
"""

from .processor import ECGProcessor
from .sqa import ECGSQAEngine

__all__ = ["ECGProcessor", "ECGSQAEngine"]
__version__ = "1.0.0"
