import numpy as np

class ECGSQAEngine:
    """Three-stage ECG signal quality assessment engine."""

    # Default thresholds — override on the instance before calling assess()
    FLAT_THRESH    = 0.50   # fraction of near-constant consecutive pairs
    NZC_THRESH     = 500    # zero-crossing count threshold for Gaussian noise
    HR_MIN         = 24     # BPM lower bound (Stage 3)
    HR_MAX         = 240    # BPM upper bound (Stage 3)
    QRS_COV_THRESH = 0.90   # max coefficient of variation of R-peak amplitudes

    def __init__(self, processor):
        """
        Parameters
        ----------
        processor : ECGProcessor
            A loaded ECGProcessor instance (from ecg_lib.processor).
        """
        self.processor = processor

    # ── Stage 1 ───────────────────────────────────────────────────────

    def check_flat_line(self):
        """Detect flat-line or saturated signal.

        Normalises the ECG to [0, 1000].  Consecutive sample pairs where
        |Δ| ≤ 1 are considered 'near-constant'.  If the fraction of such
        pairs exceeds FLAT_THRESH the signal is flat/saturated.

        Returns
        -------
        is_flat    : bool
        proportion : float   – fraction of near-constant pairs
        absdiff    : ndarray – |Δ| on the normalised signal
        """
        sig              = self.processor.filtered.copy()
        s_min, s_max     = sig.min(), sig.max()

        if s_max == s_min:
            return True, 1.0, np.zeros(max(len(sig) - 1, 0))

        sig_norm   = (sig - s_min) / (s_max - s_min) * 1000.0
        absdiff    = np.abs(np.diff(sig_norm))
        proportion = float(np.mean(absdiff <= 1.0))
        return proportion > self.FLAT_THRESH, proportion, absdiff

    # ── Stage 2 ───────────────────────────────────────────────────────

    def check_pure_noise(self):
        """Detect pure Gaussian noise via zero-crossing count.

        Scales the ECG to [-1, 1] and counts sign changes (zero crossings).
        A zero-crossing count ≥ NZC_THRESH indicates Gaussian noise.

        Returns
        -------
        is_noise : bool
        nzc      : int     – number of zero crossings
        sig_sc   : ndarray – signal scaled to [-1, 1]
        """
        sig          = self.processor.filtered.copy()
        s_min, s_max = sig.min(), sig.max()

        if s_max == s_min:
            return False, 0, np.zeros_like(sig)

        sig_sc = 2.0 * (sig - s_min) / (s_max - s_min) - 1.0
        signs  = np.sign(sig_sc)
        nz     = signs[signs != 0]
        nzc    = int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0
        return nzc >= self.NZC_THRESH, nzc, sig_sc

    # ── Stage 3 ───────────────────────────────────────────────────────

    def check_qrs(self, peaks, hr_info):
        """Check HR range and QRS amplitude coefficient of variation.

        ``hr_info`` must have been computed with the same HR_MIN / HR_MAX
        limits so that ``mean_hr`` is derived only from physiologically valid
        RR intervals.

        Parameters
        ----------
        peaks   : ndarray - sample indices of detected R-peaks
        hr_info : dict    - output of ECGProcessor.compute_hr()

        Returns
        -------
        passes  : bool
        mean_hr : float
        amp_cov : float or None
        note    : str   - human-readable reason for pass / failure
        """
        mean_hr = hr_info.get("mean_hr", 0.0)

        if len(peaks) < 2:
            return False, mean_hr, None, "Too few R-peaks detected"

        if mean_hr == 0.0:
            return False, mean_hr, None, (
                f"No valid HR — all RR intervals outside [{self.HR_MIN}–{self.HR_MAX}] BPM")

        if not (self.HR_MIN <= mean_hr <= self.HR_MAX):
            return False, mean_hr, None, (
                f"HR {mean_hr:.1f} BPM outside [{self.HR_MIN}–{self.HR_MAX}]")

        amps = np.abs(self.processor.filtered[peaks])
        mu   = np.mean(amps)
        if mu == 0:
            return False, mean_hr, None, "Zero mean QRS amplitude"

        cov = float(np.std(amps) / mu)
        if cov >= self.QRS_COV_THRESH:
            return False, mean_hr, cov, (
                f"QRS amp CoV {cov:.2f} ≥ {self.QRS_COV_THRESH}")

        return True, mean_hr, cov, "OK"

    # ── Full pipeline ─────────────────────────────────────────────────

    def assess(self, wavelet_params=None):
        """Run the three-stage feasibility pipeline sequentially.

        Stages are short-circuit: failure at any stage skips subsequent stages.
        HR is computed only from RR intervals within [HR_MIN, HR_MAX] BPM so
        that outlier peaks do not corrupt the mean HR estimate.

        Parameters
        ----------
        wavelet_params : dict, optional
            Keyword arguments forwarded to ECGProcessor.detect_peaks().
            Example: ``{"level": 6, "prominence_frac": 0.35}``

        Returns
        -------
        dict with keys:
            quality         : "acceptable" | "unacceptable"
            label           : human-readable result string
            stage_reached   : 1 | 2 | 3  (last stage evaluated)
            flat_proportion : float  - fraction of near-constant pairs (Stage 1)
            is_flat         : bool
            absdiff_norm    : ndarray - |Δ| on normalised signal (Stage 1)
            nzc             : int    - zero-crossing count (Stage 2)
            is_noise        : bool
            sig_sc          : ndarray - signal scaled to [-1, 1] (Stage 2)
            peaks           : ndarray - R-peak sample indices (Stage 3)
            qrs             : ndarray - wavelet-reconstructed QRS signal (Stage 3)
            hr_info         : dict   - from ECGProcessor.compute_hr() (Stage 3)
            mean_hr         : float  - mean HR from valid RR intervals (BPM)
            amp_cov         : float or None - QRS amplitude CoV
        """
        wp = wavelet_params or {}

        result = {
            "quality":         "unacceptable",
            "label":           "",
            "stage_reached":   0,
            "flat_proportion": None,
            "is_flat":         False,
            "absdiff_norm":    None,
            "nzc":             None,
            "is_noise":        False,
            "sig_sc":          None,
            "peaks":           None,
            "qrs":             None,
            "hr_info":         None,
            "mean_hr":         None,
            "amp_cov":         None,
        }

        # ── Stage 1: flat line / saturation ──────────────────────────
        result["stage_reached"] = 1
        is_flat, proportion, absdiff = self.check_flat_line()
        result.update(flat_proportion=proportion, is_flat=is_flat,
                      absdiff_norm=absdiff)

        if is_flat:
            result["label"] = "Unacceptable \u2013 Flat Line / Saturation"
            return result

        # ── Stage 2: pure Gaussian noise ─────────────────────────────
        result["stage_reached"] = 2
        is_noise, nzc, sig_sc = self.check_pure_noise()
        result.update(nzc=nzc, is_noise=is_noise, sig_sc=sig_sc)

        if is_noise:
            result["label"] = "Unacceptable \u2013 Pure Noise (Gaussian)"
            return result

        # ── Stage 3: QRS / HR feasibility ────────────────────────────
        result["stage_reached"] = 3
        peaks, qrs = self.processor.detect_peaks(**wp)

        # Pass HR bounds so that mean_hr is computed from valid intervals only
        hr_info = self.processor.compute_hr(
            peaks, min_bpm=self.HR_MIN, max_bpm=self.HR_MAX)
        result.update(peaks=peaks, qrs=qrs, hr_info=hr_info)

        passes, mean_hr, amp_cov, note = self.check_qrs(peaks, hr_info)
        result.update(mean_hr=mean_hr, amp_cov=amp_cov)

        if not passes:
            result["label"] = (
                f"Unacceptable \u2013 QRS / HR Check Failed\n({note})")
        else:
            result["quality"] = "acceptable"
            result["label"]   = "Acceptable"

        return result
