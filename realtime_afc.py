"""
AFC Realtime Audio - 7 Thuật toán thích nghi
==============================================

Thuật toán: NLMS, IPNLMS, APA, IPAPA, PEM-IPAPSA, PEM-MIPAPSA, PEM-BSMIPAPSA
Mỗi thuật toán sử dụng AR model (pre-whitening) và signed error convention
đồng bộ với Adpative_Filter.txt (MATLAB/Simulink).

Yêu cầu: pip install numpy pyaudio
"""

import numpy as np
import pyaudio
import time
import threading
import sys
import os
import pystoi
from scipy.signal import resample_poly, firwin2

# Uu tien dung thu vien pesq chinh thuc (ITU-T P.862.2)
# Cai dat: pip install pesq
# Neu khong co, fallback ve custom implementation
try:
    from pesq import pesq as _pesq_official
    _PESQ_OFFICIAL = True
except ImportError:
    _PESQ_OFFICIAL = False

# Allow running directly (python realtime_afc.py) or as installed package
_sys_path_dir = os.path.dirname(os.path.abspath(__file__))
if _sys_path_dir not in sys.path:
    sys.path.insert(0, _sys_path_dir)

from adaptfilt.afc_params import AFCParameters as _AFP


# ==============================================================================
# PESQ (ITU-T P.862) - Pure Python / NumPy implementation
# ==============================================================================
def _fir_filter(x, b):
    """Apply FIR filter using convolution."""
    return np.convolve(x, b, mode='full')[:len(x)]


def _resample(x, from_fs, to_fs):
    """Resample signal using linear interpolation."""
    if from_fs == to_fs:
        return x
    g = np.gcd(to_fs, from_fs)
    return resample_poly(x, to_fs // g, from_fs // g)


def _compute_irs_filter(fs):
    """Compute IRS send char. filter (ITU-T P.862, Appendix A)."""
    if fs == 8000:
        f = np.array([0, 200, 300, 350, 400, 450, 500, 570, 665,
                      770, 840, 920, 1000, 1100, 1170, 1270,
                      1400, 1520, 1630, 1750, 1900, 2050, 2150,
                      2320, 2500, 2650, 2800, 3050, 3300, 3500, 4000], dtype=np.float64)
        a = np.array([0, -200, -200, -200, -200, -200, -200, -200, -200,
                      -200, -200, -200, -200, -200, -200, -200, -200, -200, -200, -200,
                      -200, -300, -300, -400, -500, -600, -700, -900,
                      -1200, -1500, -2000], dtype=np.float64)
        H = np.power(10.0, a / 20.0)
        length = 240
    elif fs == 16000:
        f = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 570, 665,
                      770, 840, 920, 1000, 1100, 1170, 1270,
                      1400, 1520, 1630, 1750, 1900, 2050, 2150,
                      2320, 2500, 2650, 2800, 3050, 3300, 3500, 4000,
                      4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000], dtype=np.float64)
        a = np.array([0, -200, -200, -200, -200, -200, -200, -200, -200, -200, -200,
                      -200, -200, -200, -200, -200, -200, -200, -200, -200,
                      -200, -300, -300, -400, -500, -600, -700, -900,
                      -1200, -1500, -2000, -2500, -3000, -3500, -4000,
                      -5000, -6000, -7000, -8000, -9000, -10000, -11000, -12000],
                     dtype=np.float64)
        H = np.power(10.0, a / 20.0)
        length = 480
    else:
        return np.array([1.0])
    b = firwin2(length, f, H, fs=fs)
    return b


def _window_frame(sig, frame_len, hop):
    """Window signal into overlapping frames."""
    n_frames = max(1, (len(sig) - frame_len) // hop + 1)
    frames = np.zeros((n_frames, frame_len), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        frames[i, :min(frame_len, len(sig) - start)] = sig[start:min(end, len(sig))]
    return frames


def _bark_band_powers(sig, fs, n_bark=23, n_fft=512):
    """Compute per-frame Bark band powers from signal."""
    frame_len = int(0.03 * fs)
    hop = frame_len // 2
    frames = _window_frame(sig, frame_len, hop)
    if len(frames) < 3:
        return np.zeros((3, n_bark))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    bark = 13.0 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)

    band_edges = np.zeros(n_bark + 1, dtype=np.int32)
    band_edges[0] = 0
    for b in range(1, n_bark):
        diffs = np.abs(bark - float(b))
        band_edges[b] = int(np.argmin(diffs)) + 1
    band_edges[n_bark] = len(freqs)

    bark_powers = np.zeros((len(frames), n_bark), dtype=np.float64)
    for i in range(len(frames)):
        X = np.fft.rfft(frames[i] * np.hamming(frame_len), n=n_fft)
        for b in range(n_bark):
            start = max(0, band_edges[b] - 1)
            end = min(len(freqs), band_edges[b + 1])
            bark_powers[i, b] = np.sum(np.abs(X[start:end]) ** 2) + 1e-12

    return bark_powers


def compute_pesq(ref, deg, fs):
    """
    Compute PESQ-like score (ITU-T P.862 inspired).
    Uses Bark-domain power difference with Zwicker loudness.

    Parameters
    ----------
    ref : array_like
        Reference (clean) signal.
    deg : array_like
        Degraded signal.
    fs : int
        Sampling frequency (8000 or 16000 Hz).

    Returns
    -------
    float
        PESQ score in range [-0.5, 4.5] (MOS-LQO scale).
        Returns 0.0 if signal too short (< 0.5s).
    """
    ref = np.asarray(ref, dtype=np.float64)
    deg = np.asarray(deg, dtype=np.float64)

    if len(ref) < fs // 2 or len(deg) < fs // 2:
        return 0.0

    if fs not in (8000, 16000):
        fs_target = 16000 if fs >= 16000 else 8000
        ref = _resample(ref, fs, fs_target)
        deg = _resample(deg, fs, fs_target)
        fs = fs_target

    ref_p = ref / (np.max(np.abs(ref)) + 1e-12)
    deg_p = deg / (np.max(np.abs(deg)) + 1e-12)

    irs_b = _compute_irs_filter(fs)
    ref_f = _fir_filter(ref_p, irs_b)
    deg_f = _fir_filter(deg_p, irs_b)

    ref_bark = _bark_band_powers(ref_f, fs)
    deg_bark = _bark_band_powers(deg_f, fs)

    n_frames = min(len(ref_bark), len(deg_bark))
    ref_bark = ref_bark[:n_frames]
    deg_bark = deg_bark[:n_frames]

    if n_frames < 5:
        return 0.0

    # Bark-domain power difference (in dB)
    ref_db = 10.0 * np.log10(ref_bark)
    deg_db = 10.0 * np.log10(deg_bark)
    diff_db = deg_db - ref_db

    # Zwicker loudness: L = sum_j(S_j^0.3) per frame
    ref_loud = np.sum(np.power(ref_bark, 0.3), axis=1, keepdims=True) + 1e-12
    deg_loud = np.sum(np.power(deg_bark, 0.3), axis=1, keepdims=True) + 1e-12

    # Asymmetric disturbance (P.862 eq. 1-2): penalize added noise more than missing signal
    alpha = 0.5
    diff_loud = deg_loud - ref_loud
    D_plus = np.maximum(diff_loud, 0.0)
    D_minus = np.minimum(diff_loud, 0.0)
    D_asym = D_plus + alpha * D_minus

    # Per-band asymmetric disturbance weighted by relative band power
    band_weights = ref_bark / (np.sum(ref_bark, axis=1, keepdims=True) + 1e-12)
    D_asym_weighted = D_asym * band_weights

    # Frame-level: sum over bands, then time-filter
    D_frame = np.sum(D_asym_weighted, axis=1)

    # 3-pass forward IIR smoothing
    for _ in range(3):
        D_filt = np.zeros_like(D_frame)
        D_filt[0] = D_frame[0]
        for i in range(1, len(D_frame)):
            D_filt[i] = 0.5 * D_filt[i - 1] + 0.5 * D_frame[i]
        D_frame = D_filt

    D_frame = np.clip(D_frame, 0, 5)
    D_mean = float(np.mean(D_frame))
    D_max = float(np.max(D_frame))

    # PESQ MOS-LQO mapping
    pesq_score = 4.5 - 3.0 * np.arctan(3.0 * D_mean + 0.5 * D_max)

    return float(np.clip(pesq_score, -0.5, 4.5))


def compute_snr_seg(ref, deg, fs, frame_dur=0.03, overlap=0.5):
    """
    Compute Segmental SNR (SNRseg) in dB.

    Parameters
    ----------
    ref, deg : array_like
        Reference and degraded signals.
    fs : int
        Sampling frequency.
    frame_dur : float
        Frame duration in seconds (default 30ms).
    overlap : float
        Frame overlap ratio (default 0.5 = 50%%).

    Returns
    -------
    float
        SNRseg in dB. Higher = better quality.
        Typical range: -10 to +35 dB.
    """
    ref = np.asarray(ref, dtype=np.float64).flatten()
    deg = np.asarray(deg, dtype=np.float64).flatten()
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    frame_len = int(frame_dur * fs)
    hop = int(frame_len * (1 - overlap))
    n_frames = max(1, (min_len - frame_len) // hop + 1)

    snr_db_vals = []
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > min_len:
            break
        r = ref[start:end]
        d = deg[start:end]

        P_signal = np.sum(r ** 2) + 1e-12
        P_noise = np.sum((d - r) ** 2) + 1e-12
        snr_db = 10.0 * np.log10(P_signal / P_noise)
        snr_db_vals.append(snr_db)

    if not snr_db_vals:
        return 0.0
    snr_db_vals = np.array(snr_db_vals)

    # Discard extreme frames (below -10 dB likely silence)
    valid = (snr_db_vals > -10) & (snr_db_vals < 35)
    if np.sum(valid) == 0:
        return float(np.mean(snr_db_vals))
    return float(np.mean(snr_db_vals[valid]))


def compute_lsd(ref, deg, fs, n_fft=512):
    """
    Compute Log-Spectral Distance (LSD) in dB.

    Parameters
    ----------
    ref, deg : array_like
        Reference and degraded signals.
    fs : int
        Sampling frequency.

    Returns
    -------
    float
        Mean LSD in dB. Lower = better quality.
        Typical range: 0 to 10 dB (0 = identical).
    """
    ref = np.asarray(ref, dtype=np.float64).flatten()
    deg = np.asarray(deg, dtype=np.float64).flatten()
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    hop = n_fft // 4
    n_frames = max(1, (min_len - n_fft) // hop)

    lsd_vals = []
    for i in range(n_frames):
        start = i * hop
        end = start + n_fft
        if end > min_len:
            break
        R = np.fft.rfft(ref[start:end] * np.hamming(n_fft), n=n_fft)
        D = np.fft.rfft(deg[start:end] * np.hamming(n_fft), n=n_fft)
        power_ref = np.abs(R) ** 2 + 1e-12
        power_deg = np.abs(D) ** 2 + 1e-12
        log_diff = (10.0 / np.log(10.0)) * np.log(power_deg / power_ref)
        lsd = np.sqrt(np.mean(log_diff ** 2))
        if np.isfinite(lsd) and lsd < 20:
            lsd_vals.append(lsd)

    if not lsd_vals:
        return 0.0
    return float(np.mean(lsd_vals))


# ------------------------------------------------------------------
# Levinson-Durbin recursion cho AR model
# ------------------------------------------------------------------
def levinson_durbin(r, order, max_coeff=6.0):
    """
    Levinson-Durbin recursion.
    r = autocorrelation vector [R[0], R[1], ..., R[order]] (already normalized by R[0] externally)
    order = AR model order
    max_coeff : float
        AR coefficient clamp limit — should match AFCParameters.max_ar (default 6).
        Previously hardcoded as 10.0, which was inconsistent with AFCParameters.max_ar = 6.
    Returns: AR coefficients [1, a1, a2, ..., a_order], prediction error E
    """
    if order == 0:
        return np.array([1.0]), r[0]

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0

    E = r[0]
    if not np.isfinite(E) or abs(E) < 1e-12:
        E = 1e-12

    for p in range(1, order + 1):
        num = sum(r[j] * a[j] for j in range(1, p + 1))
        k = (r[p] - num) / E

        # Clamp reflection coefficient to [-1, 1] for stability
        if not np.isfinite(k):
            k = 0.0
        k = max(-1.0, min(1.0, k))

        for j in range(1, p):
            a[j] = a[j] + k * a[p - j]
        a[p] = k

        E = E * (1.0 - k * k)
        if not np.isfinite(E) or abs(E) < 1e-12:
            E = 1e-12

    # Clamp all coefficients using the passed max_coeff (= self.max_ar = AFCParameters.max_ar)
    for j in range(1, order + 1):
        if not np.isfinite(a[j]) or abs(a[j]) > max_coeff:
            a[j] = max(-max_coeff, min(max_coeff, a[j]))

    return a, E


def ar_filter(sample, ar_coeffs, delay_line):
    """Lọc FIR một mẫu qua AR model: y = sum(ar_coeffs[i] * delay_line[i])."""
    delay_line[1:] = delay_line[:-1]
    delay_line[0] = sample
    if len(ar_coeffs) == 0:
        return 0.0, delay_line
    n = min(len(ar_coeffs), len(delay_line))
    y = sum(ar_coeffs[j] * delay_line[j] for j in range(n))
    # Clamp output to prevent overflow
    if not np.isfinite(y):
        y = 0.0
    return y, delay_line


def sign(x):
    """Hard signed error: sign(0) = 1 (đồng bộ MATLAB convention)."""
    return np.where(x >= 0, 1.0, -1.0)


def soft_sign(x, lda):
    """
    Soft signed error: tanh(lda * x).
    Đồng bộ MATLAB/Simulink: lda = AFCParameters.lda = 6.
    Approaches hard sign as lda → ∞.
    Previously lda was stored in self.lda but sign() was always called without lda,
    so the parameter had zero effect. Now used throughout all algorithm update steps.
    """
    return np.tanh(lda * x)


# ------------------------------------------------------------------
# AdaptiveFilter - 7 thuật toán thích nghi
# ------------------------------------------------------------------
class AdaptiveFilter:
    """
    Bộ lọc thích nghi AFC với 7 thuật toán.
    Đồng bộ với Adpative_Filter.txt (MATLAB/Simulink).
    """

    ALGORITHMS = {
        'nlms':     'PEM-NLMS',
        'ipnlms':   'PEM-IPNLMS',
        'apa':      'PEM-APA',
        'ipapa':    'PEM-IPAPA',
        'ipapsa':   'PEM-IPAPSA',
        'mipapsa':  'PEM-MIPAPSA',
        'bsmipapsa': 'PEM-BSMIPAPSA',
    }

    def __init__(self, M=64, step=None, delta=1e-6,
                 a=0.5, p=2, algo='nlms',
                 w_max_norm=None, leaky=0.0,
                 # ---- MATLAB/Simulink AFC parameters ----
                 mu1=4e-6, mu2=8e-2, eps=1e-5, lda=6,
                 delta_sc=1e-8, beta=50, M_bs=8,
                 tanh_scale=0.5, tanh_thresh=0.15,
                 max_ar=6, fs=16000, d_k=96, d_fb=1):
        if algo not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algo}. Choose from: {list(self.ALGORITHMS.keys())}")

        # APA-family data matrices (MAX_P must be set before p validation)
        self.MAX_P = 10

        if p > self.MAX_P:
            raise ValueError(f"Projection order p={p} exceeds MAX_P={self.MAX_P}. Reduce p or recompile with larger MAX_P.")

        self.M = M
        # mu is algorithm-dependent; use AFCParameters defaults when step not specified
        if step is None:
            self.mu = _AFP.get_mu(algo)
        else:
            self.mu = step
        self.delta = delta
        self.a = a
        self.p = p
        self.algo = algo
        self.algo_name = self.ALGORITHMS[algo]
        self.w_max_norm = w_max_norm if w_max_norm is not None else float('inf')
        self.leaky = leaky

        # Feedback path filter coefficients (gTD)
        self.gTD = np.zeros(M, dtype=np.float64)

        # Delay lines
        self.TDLLs = np.zeros(M, dtype=np.float64)
        self.TDLLswh = np.zeros(M, dtype=np.float64)

        # APA-family data matrices
        self.TDLMicwh = np.zeros(self.MAX_P, dtype=np.float64)
        self.TDLLswh_d = np.zeros(M + self.MAX_P - 1, dtype=np.float64)
        self.Lswh_ap = np.zeros((M, self.MAX_P), dtype=np.float64)

        # PEM-MIPAPSA / PEM-BSMIPAPSA memory
        self.Q_tilde_prev = np.zeros((M, self.MAX_P), dtype=np.float64)

        # AP approximation params

        # --- Algorithm Parameters (from MATLAB/Simulink) ---
        self.mu1 = mu1      # NLMS step when stable (sw2 mode): mu/2 = 4e-6
        self.mu2 = mu2      # NLMS step when unstable: mu*10^4 = 8e-2
        self.eps = eps      # Epsilon in regression matrix: R_mu = eps*I
        self.lda = lda       # Signed error sigmoid coefficient
        self.delta_sc = delta_sc  # Sparseness measure regularization
        self.beta = beta     # TanH-gated threshold factor (beta/1000)
        self.M_bs = M_bs    # Number of blocks for BSMIPAPSA
        self.tanh_scale = tanh_scale  # Error clipping: e = 2*tanh(0.5*e)
        self.tanh_thresh = tanh_thresh  # Impulse detection threshold

        # --- System Parameters ---
        self.fs = fs        # Sampling frequency (Hz)
        self.d_k = d_k      # Forward path delay (samples)
        self.d_fb = d_fb    # Feedback cancellation path delay (samples)

        # AR model (pre-whitening) — MATLAB: AR_ORDER=La=20, FRAMELENGTH=framelength=160
        self.AR_ORDER = _AFP.La           # = 20 samples (from AFCParameters)
        self.FRAMELENGTH = _AFP.framelength  # = 160 samples = 10ms at 16kHz
        self.max_ar = max_ar  # AR coefficient clamp limit (was 5.0, now 6)
        self.ar_coeffs = np.zeros(self.AR_ORDER, dtype=np.float64)
        self.ar_coeffs[0] = 1.0
        self.ar_frame = np.zeros(self.FRAMELENGTH, dtype=np.float64)
        self.ar_frameindex = 0
        self.ar_delay_mic = np.zeros(self.AR_ORDER, dtype=np.float64)
        self.ar_delay_ls = np.zeros(self.AR_ORDER, dtype=np.float64)
        self.mic_delay_buf = np.zeros(self.FRAMELENGTH + 1, dtype=np.float64)
        self.ls_delay_buf = np.zeros(self.FRAMELENGTH + 1, dtype=np.float64)

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core processing per sample
    # ------------------------------------------------------------------
    def process_sample(self, mic_sample, ref_sample):
        """
        Xử lý một mẫu (đồng bộ MATLAB: gọi trong vòng lặp mỗi mẫu).
        Returns: (error, feedback_estimate)
        """
        with self._lock:
            # 1. Cập nhật TDLLs (reference signal delay line)
            self.TDLLs[1:] = self.TDLLs[:-1]
            self.TDLLs[0] = ref_sample

            # 2. Tính error signal: e = mic - gTD^T * TDLLs
            e = mic_sample - np.dot(self.gTD, self.TDLLs)
            if not np.isfinite(e):
                e = 0.0

            # 3. Clipping: e = 2 * tanh(scale * e) — synchronized with MATLAB
            e_clipped = 2.0 * np.tanh(self.tanh_scale * e)

            # 4. Delay & AR filter signals
            # Delay mic signal
            self.mic_delay_buf[1:] = self.mic_delay_buf[:-1]
            self.mic_delay_buf[0] = mic_sample
            Micdelay = self.mic_delay_buf[self.FRAMELENGTH]

            # Delay ref signal
            self.ls_delay_buf[1:] = self.ls_delay_buf[:-1]
            self.ls_delay_buf[0] = ref_sample
            Lsdelay = self.ls_delay_buf[self.FRAMELENGTH]

            # AR filter (pre-whitening)
            Micwh, self.ar_delay_mic = ar_filter(Micdelay, self.ar_coeffs, self.ar_delay_mic)
            Lswh, self.ar_delay_ls = ar_filter(Lsdelay, self.ar_coeffs, self.ar_delay_ls)
            if not np.isfinite(Micwh):
                Micwh = 0.0
            if not np.isfinite(Lswh):
                Lswh = 0.0

            # 5. Update AR model frame & Levinson
            self.ar_frame[1:] = self.ar_frame[:-1]
            self.ar_frame[0] = e_clipped

            if self.ar_frameindex == self.FRAMELENGTH - 1 and self.AR_ORDER - 1 > 0:
                R = np.zeros(self.AR_ORDER, dtype=np.float64)
                for j in range(self.AR_ORDER):
                    vec_mult = np.zeros(self.FRAMELENGTH, dtype=np.float64)
                    end_idx = min(self.FRAMELENGTH, self.FRAMELENGTH - j)
                    vec_mult[:end_idx] = self.ar_frame[j:j + end_idx]
                    R[j] = np.dot(self.ar_frame, vec_mult) / self.FRAMELENGTH

                # Normalize autocorrelation by R[0] for Levinson-Durbin
                R0 = R[0]
                if abs(R0) > 1e-12:
                    R_norm = R / R0
                else:
                    R_norm = R.copy()
                    R_norm[0] = 1.0

                ar_new, _ = levinson_durbin(R_norm, self.AR_ORDER - 1, max_coeff=self.max_ar)
                self.ar_coeffs.fill(0.0)
                n_copy = min(len(ar_new), self.AR_ORDER)
                self.ar_coeffs[:n_copy] = ar_new[:n_copy]

                # Clamp AR coefficients to prevent instability
                max_ar = self.max_ar
                for j in range(len(self.ar_coeffs)):
                    if not np.isfinite(self.ar_coeffs[j]):
                        self.ar_coeffs[j] = 0.0
                    self.ar_coeffs[j] = max(-max_ar, min(max_ar, self.ar_coeffs[j]))

            self.ar_frameindex = (self.ar_frameindex + 1) % self.FRAMELENGTH

            # 6. Cập nhật TDLLswh (pre-whitened signal delay line)
            self.TDLLswh[1:] = self.TDLLswh[:-1]
            self.TDLLswh[0] = Lswh

            # 7. Tính ep = Micwh - Lswh^T * gTD
            ep = Micwh - np.dot(self.TDLLswh, self.gTD)
            if not np.isfinite(ep):
                ep = 0.0

            # 8. Cập nhật APA data matrices
            self.TDLMicwh[1:] = self.TDLMicwh[:-1]
            self.TDLMicwh[0] = Micwh

            self.TDLLswh_d[1:] = self.TDLLswh_d[:-1]
            self.TDLLswh_d[0] = Lswh

            for i in range(self.p):
                vec_start = i
                vec_end = i + self.M - 1
                if vec_end < len(self.TDLLswh_d):
                    self.Lswh_ap[:, i] = self.TDLLswh_d[vec_start:vec_end + 1]
                else:
                    avail = len(self.TDLLswh_d) - vec_start
                    col = np.zeros(self.M, dtype=np.float64)
                    col[:avail] = self.TDLLswh_d[vec_start:]
                    self.Lswh_ap[:, i] = col

            # Active submatrices
            Lswh_ap_active = self.Lswh_ap[:, :self.p]
            TDLMicwh_active = self.TDLMicwh[:self.p]
            ewh_p = TDLMicwh_active - Lswh_ap_active.T @ self.gTD

            # ------------------------------------------------------------------
            # 9. Thuật toán thích nghi
            # ------------------------------------------------------------------
            norm_reg = (1.0 - self.a) / (2.0 * self.M) * self.delta
            denom_norm = 0.0

            if self.algo == 'nlms':
                norm_sq = np.dot(self.TDLLswh, self.TDLLswh) + self.delta
                u_pgs = self.TDLLswh * soft_sign(ep, self.lda)
                denom_norm = np.dot(u_pgs, u_pgs)
                update = self.mu * u_pgs / (norm_sq + norm_reg)

            elif self.algo == 'ipnlms':
                b = (1.0 - self.a) / (2.0 * self.M) + \
                    (1.0 + self.a) * np.abs(self.gTD) / (np.sum(np.abs(self.gTD)) + self.delta_sc)
                u_pgs = b * self.TDLLswh * soft_sign(ep, self.lda)
                denom_norm = np.dot(u_pgs, u_pgs)
                update = self.mu * u_pgs / (denom_norm + norm_reg)

            elif self.algo == 'apa':
                AtA = Lswh_ap_active.T @ Lswh_ap_active
                A_mat = self.delta * np.eye(self.p, dtype=np.float64) + AtA
                rc = np.linalg.cond(A_mat)
                if not np.isfinite(rc) or rc > 1e10:
                    A_mat += (max(self.delta, 1e-6) * 10) * np.eye(self.p, dtype=np.float64)
                update = Lswh_ap_active @ np.linalg.solve(A_mat, soft_sign(ewh_p, self.lda))
                update = self.mu * update

            elif self.algo == 'ipapa':
                b = (1.0 - self.a) / (2.0 * self.M) + \
                    (1.0 + self.a) * np.abs(self.gTD) / (np.sum(np.abs(self.gTD)) + self.delta_sc)
                B = np.diag(b)
                u_pgs = B @ Lswh_ap_active @ soft_sign(ewh_p, self.lda)
                denom_norm = np.dot(u_pgs, u_pgs)
                update = self.mu * u_pgs / (denom_norm + norm_reg)

            elif self.algo == 'ipapsa':
                b = (1.0 - self.a) / (2.0 * self.M) + \
                    (1.0 + self.a) * np.abs(self.gTD) / (np.sum(np.abs(self.gTD)) + self.delta_sc)
                B = np.diag(b)
                u_pgs = B @ Lswh_ap_active @ soft_sign(ewh_p, self.lda)
                denom_norm = np.dot(u_pgs, u_pgs)
                update = self.mu * u_pgs / (denom_norm + norm_reg)

            elif self.algo == 'mipapsa':
                b = (1.0 - self.a) / (2.0 * self.M) + \
                    (1.0 + self.a) * np.abs(self.gTD) / (np.sum(np.abs(self.gTD)) + self.delta_sc)
                current_col = b * Lswh_ap_active[:, 0]

                if self.p >= 2:
                    Q_tilde = np.zeros((self.M, self.p), dtype=np.float64)
                    Q_tilde[:, 0] = current_col
                    cols_from_prev = min(self.p - 1, self.Q_tilde_prev.shape[1])
                    Q_tilde[:, 1:1 + cols_from_prev] = self.Q_tilde_prev[:, :cols_from_prev]
                else:
                    Q_tilde = current_col

                u_tilde_pgs = Q_tilde @ soft_sign(ewh_p, self.lda)
                denom_norm = np.dot(u_tilde_pgs, u_tilde_pgs)
                update = self.mu * u_tilde_pgs / (denom_norm + norm_reg)

                if self.p >= 2:
                    self.Q_tilde_prev[:, 1:] = self.Q_tilde_prev[:, :-1]
                    self.Q_tilde_prev[:, 0] = current_col

            elif self.algo == 'bsmipapsa':
                M_bs = self.M_bs
                N_bs = self.M // M_bs
                b_hat = np.zeros(self.M, dtype=np.float64)
                block_norms = np.zeros(M_bs, dtype=np.float64)

                for blk in range(M_bs):
                    idx_start = blk * N_bs
                    idx_end = idx_start + N_bs
                    block_norms[blk] = np.linalg.norm(self.gTD[idx_start:idx_end])

                sum_block_norms = np.sum(block_norms)

                for blk in range(M_bs):
                    idx_start = blk * N_bs
                    idx_end = idx_start + N_bs
                    b_hat_k = (1.0 - self.a) / (2.0 * self.M) + \
                              (1.0 + self.a) * block_norms[blk] / (2.0 * M_bs * sum_block_norms + self.delta_sc)
                    b_hat[idx_start:idx_end] = b_hat_k

                current_col = b_hat * Lswh_ap_active[:, 0]

                if self.p >= 2:
                    Q_hat = np.zeros((self.M, self.p), dtype=np.float64)
                    Q_hat[:, 0] = current_col
                    cols_from_prev = min(self.p - 1, self.Q_tilde_prev.shape[1])
                    Q_hat[:, 1:1 + cols_from_prev] = self.Q_tilde_prev[:, :cols_from_prev]
                else:
                    Q_hat = current_col

                u_hat_pgs = Q_hat @ soft_sign(ewh_p, self.lda)
                denom_norm = np.dot(u_hat_pgs, u_hat_pgs)
                update = self.mu * u_hat_pgs / (denom_norm + norm_reg)

                if self.p >= 2:
                    self.Q_tilde_prev[:, 1:] = self.Q_tilde_prev[:, :-1]
                    self.Q_tilde_prev[:, 0] = current_col

            # Leaky update: w(n+1) = (1 - leaky)*w(n) + update
            # Leaky LMS must DECAY the weight vector to prevent blow-up.
            if self.leaky > 0:
                self.gTD = (1.0 - self.leaky) * self.gTD + update
            else:
                # Apply update
                self.gTD = self.gTD + update

            # DC removal on gTD is DISABLED. Reason: mean(gTD) ~ 0.039/step
            # while adaptive update ~ 9e-5/step → DC removal kills weights faster than
            # the algorithm can learn, causing W_norm to drop while RMS rises (diverging).
            # The acoustic feedback path has a real DC component that should not be removed.

            # Guard NaN in weights
            for j in range(self.M):
                if not np.isfinite(self.gTD[j]):
                    self.gTD[j] = 0.0

            # Weight norm clamping
            w_norm = np.linalg.norm(self.gTD)
            if np.isfinite(w_norm) and w_norm > self.w_max_norm and self.w_max_norm != float('inf'):
                self.gTD = self.gTD * (self.w_max_norm / w_norm)

            # Estimate feedback
            y = np.dot(self.gTD, self.TDLLs)
            if not np.isfinite(y):
                y = 0.0

            return e, y

    def process_chunk(self, mic_chunk, ref_chunk):
        """Process a chunk of samples (wrapper around per-sample processing)."""
        n = len(mic_chunk)
        errors = np.zeros(n, dtype=np.float32)
        y_est = np.zeros(n, dtype=np.float32)

        for i in range(n):
            e, y = self.process_sample(float(mic_chunk[i]), float(ref_chunk[i]))
            errors[i] = e
            y_est[i] = y

        return errors.astype(np.float32), y_est.astype(np.float32)

    def reset(self):
        with self._lock:
            self.gTD.fill(0.0)
            self.TDLLs.fill(0.0)
            self.TDLLswh.fill(0.0)
            self.TDLMicwh.fill(0.0)
            self.TDLLswh_d.fill(0.0)
            self.Lswh_ap.fill(0.0)
            self.Q_tilde_prev.fill(0.0)
            self.ar_coeffs.fill(0.0)
            self.ar_coeffs[0] = 1.0
            self.ar_frame.fill(0.0)
            self.ar_frameindex = 0
            self.ar_delay_mic.fill(0.0)
            self.ar_delay_ls.fill(0.0)
            self.mic_delay_buf.fill(0.0)
            self.ls_delay_buf.fill(0.0)

    def switch_mu(self, stable):
        """Switch between mu1 (stable) and mu2 (unstable) for HNLMS recovery.

        When acoustic feedback causes instability, the error magnitude spikes.
        Switching to mu2 (large step) helps the filter converge quickly to the
        new feedback path. Once stable, mu1 (small step) ensures fine tracking.
        """
        with self._lock:
            self.mu = self.mu2 if not stable else self.mu1

    def get_stats(self):
        with self._lock:
            return {
                'w_norm': float(np.linalg.norm(self.gTD)),
                'w_max': float(np.max(np.abs(self.gTD))),
                'w_mean': float(np.mean(np.abs(self.gTD))),
                'algo': self.algo_name,
                'mu': self.mu,
                'mu1': self.mu1,
                'mu2': self.mu2,
                'delta': self.delta,
                'eps': self.eps,
                'a': self.a,
                'p': self.p,
                'M_bs': self.M_bs,
                'tanh_scale': self.tanh_scale,
                'tanh_thresh': self.tanh_thresh,
                'lda': self.lda,
                'beta': self.beta,
                'delta_sc': self.delta_sc,
                'max_ar': self.max_ar,
                'fs': self.fs,
                'd_k': self.d_k,
                'd_fb': self.d_fb,
            }


# ------------------------------------------------------------------
# AFCManager - Audio I/O
# ------------------------------------------------------------------
class AFCManager:
    MODE_FULL_DUPLEX = 'full_duplex'
    MODE_LOOPBACK = 'loopback'

    def __init__(self, afc, sample_rate=44100, chunk_size=512,
                 stoi_win_sec=1.0, pesq_win_sec=4.0):
        self.afc = afc
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.pyaudio = pyaudio.PyAudio()
        self.running = False
        self.mode = None

        self.mic_buffer = np.zeros(chunk_size, dtype=np.float32)
        self.ref_buffer = np.zeros(chunk_size, dtype=np.float32)
        # Hearing-aid amplification gain K (default 30 dB from AFCParameters)
        # out = K * (mic - feedback_estimate) = K * error
        self.output_gain = 1.0   # will be set externally via manager.output_gain = K
        self._smooth_alpha = 0.1
        self._clipped_ratio = 0.0

        # AGC state
        self._agc_enabled = True

        # ── Startup ramp: gain goes 0 → target over _ramp_duration_sec ──
        # Reason: AFC filter starts at zero → no feedback estimate.
        # If gain > 0 from chunk 1, the speaker output feeds back into mic,
        # amplifies, and RMS explodes (153 → 241 billion in 1 chunk observed).
        # Ramp gives the adaptive filter time to learn the feedback path
        # before the full hearing-aid gain is applied.
        self._ramp_duration_sec = 5.0
        self._ramp_start_time = None
        self._ramp_done = False
        self._ramp_factor = 0.0

        # Stability detection for mu1/mu2 switching (HNLMS recovery)
        # Uses error RMS relative to a slowly-tracked "expected" level.
        # When error exceeds threshold, switch to mu2 (fast convergence).
        # When error returns below threshold, switch back to mu1 (fine tracking).
        self._stable = True
        self._stable_count = 0
        self._stable_threshold = 0.20   # switch to mu2 when error RMS > 20% of expected
        self._stable_hysteresis = 0.10   # stay at mu2 until error RMS drops below 10% of expected
        self._stable_min_frames = 100    # at least this many consecutive stable frames before switching to mu1
        self._error_rms_history = np.zeros(64, dtype=np.float64)  # circular buffer for RMS history
        self._error_rms_head = 0
        self._error_rms_count = 0
        self._expected_rms = 0.01        # initial expected RMS (conservative)

        self.stats = {
            'samples_processed': 0,
            'overflow_count': 0,
        }
        self._lock = threading.Lock()
        self._print_lock = threading.Lock()

        # STOI buffers
        self._stoi_buffer_len = int(stoi_win_sec * sample_rate)
        self._stoi_clean = np.zeros(self._stoi_buffer_len, dtype=np.float64)
        self._stoi_processed = np.zeros(self._stoi_buffer_len, dtype=np.float64)
        self._stoi_clean_head = 0
        self._stoi_ready = False
        self._stoi_win_sec = stoi_win_sec
        self._stoi_last_compute = 0.0
        self._current_stoi = 0.0

        # PESQ buffers (longer window: ~4s)
        self._pesq_buffer_len = int(pesq_win_sec * sample_rate)
        self._pesq_clean = np.zeros(self._pesq_buffer_len, dtype=np.float64)
        self._pesq_processed = np.zeros(self._pesq_buffer_len, dtype=np.float64)
        self._pesq_clean_head = 0
        self._pesq_ready = False
        self._pesq_win_sec = pesq_win_sec
        self._pesq_last_compute = 0.0
        self._current_pesq = 0.0
        self._current_snrseg = 0.0
        self._current_lsd = 0.0

    def _list_devices(self):
        devices = []
        info = self.pyaudio.get_host_api_info_by_index(0)
        for i in range(info.get('deviceCount', 0)):
            try:
                d = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                devices.append({
                    'index': i,
                    'name': d.get('name', ''),
                    'inputs': d.get('maxInputChannels', 0),
                    'outputs': d.get('maxOutputChannels', 0),
                })
            except Exception:
                pass
        return devices

    def _print(self, msg, end='\n', flush=True):
        with self._print_lock:
            if end == '\r':
                sys.stdout.write('\r' + msg + ' ' * 30)
                sys.stdout.flush()
            else:
                print(msg, end=end, flush=flush)

    def _push_stoi(self, clean_sample, processed_sample):
        """Append sample to circular STOI buffers."""
        buf_len = self._stoi_buffer_len
        self._stoi_clean[self._stoi_clean_head] = clean_sample
        self._stoi_processed[self._stoi_clean_head] = processed_sample
        self._stoi_clean_head = (self._stoi_clean_head + 1) % buf_len
        if self._stoi_clean_head == 0:
            self._stoi_ready = True

    def _compute_stoi(self):
        """Compute STOI from circular buffers. Returns (stoi_val, n_frames)."""
        if not self._stoi_ready:
            return 0.0, 0
        buf_len = self._stoi_buffer_len
        if self._stoi_clean_head == 0:
            clean_seg = self._stoi_clean
            proc_seg = self._stoi_processed
        else:
            clean_seg = np.roll(self._stoi_clean, -self._stoi_clean_head)
            proc_seg = np.roll(self._stoi_processed, -self._stoi_clean_head)
        try:
            stoi_val = pystoi.stoi(clean_seg, proc_seg, self.sample_rate, extended=False)
        except Exception:
            stoi_val = 0.0
        return float(stoi_val)

    def _push_pesq(self, clean_sample, processed_sample):
        """Append sample to circular PESQ buffers."""
        buf_len = self._pesq_buffer_len
        self._pesq_clean[self._pesq_clean_head] = clean_sample
        self._pesq_processed[self._pesq_clean_head] = processed_sample
        self._pesq_clean_head = (self._pesq_clean_head + 1) % buf_len
        if self._pesq_clean_head == 0:
            self._pesq_ready = True

    def _compute_pesq(self):
        """Compute PESQ, SNRseg, and LSD from circular buffers.
        Su dung thu vien pesq chinh thuc (ITU-T P.862.2) neu co,
        fallback ve custom Bark-domain implementation neu khong.
        """
        if not self._pesq_ready:
            return 0.0, 0.0, 0.0
        buf_len = self._pesq_buffer_len
        if self._pesq_clean_head == 0:
            clean_seg = self._pesq_clean
            proc_seg = self._pesq_processed
        else:
            clean_seg = np.roll(self._pesq_clean, -self._pesq_clean_head)
            proc_seg = np.roll(self._pesq_processed, -self._pesq_clean_head)
        try:
            if _PESQ_OFFICIAL:
                # ITU-T P.862.2 wideband (16kHz) hoac narrowband (8kHz)
                mode = 'wb' if self.sample_rate == 16000 else 'nb'
                pesq_result = _pesq_official(self.sample_rate,
                                            clean_seg.astype(np.float32),
                                            proc_seg.astype(np.float32),
                                            mode)
                # pesq library returns PesqResult(nb_peaq_score, ...) or PesqResult(pesq, N_del)
                # Extract the score regardless of result type (namedtuple, tuple, or float)
                try:
                    pesq_val = float(getattr(pesq_result, 'pesq',
                                           pesq_result[0] if hasattr(pesq_result, '__getitem__') else pesq_result))
                except Exception:
                    pesq_val = float(pesq_result)
            else:
                pesq_val = compute_pesq(clean_seg, proc_seg, self.sample_rate)
        except Exception:
            pesq_val = 0.0
        try:
            snrseg_val = compute_snr_seg(clean_seg, proc_seg, self.sample_rate)
        except Exception:
            snrseg_val = 0.0
        try:
            lsd_val = compute_lsd(clean_seg, proc_seg, self.sample_rate)
        except Exception:
            lsd_val = 0.0
        return float(pesq_val), float(snrseg_val), float(lsd_val)

    # _apply_agc() was removed — gain management is fully inline in run_full_duplex/run_loopback.
    # Previous RMS-based AGC (slow envelope follower) conflicted with the faster
    # clip-driven gain adjustment, causing double-adjustment and erratic behavior.
    # Inline AGC is simpler, more predictable, and easier to tune.

    def _reset_startup_ramp(self):
        """Reset ramp state so a fresh ramp runs when streams restart."""
        self._ramp_start_time = None
        self._ramp_done = False
        self._ramp_factor = 0.0

    def _apply_startup_ramp(self, now):
        """Apply startup ramp: gain ramps from 0 to target over _ramp_duration_sec.

        This prevents feedback blow-up at startup. When ramp is active, the
        hearing-aid gain is held at 0 (muted) while the AFC filter learns the
        feedback path. Once ramp completes, full gain is available for AGC.
        """
        if self._ramp_done:
            return

        if self._ramp_start_time is None:
            self._ramp_start_time = now

        elapsed = now - self._ramp_start_time
        if elapsed >= self._ramp_duration_sec:
            self._ramp_done = True
            self._ramp_factor = 1.0
        else:
            self._ramp_factor = elapsed / self._ramp_duration_sec

    def run_full_duplex(self, mic_idx, spk_idx, ref_idx=None):
        self.mode = self.MODE_FULL_DUPLEX
        use_monitor = (ref_idx is None)

        self._print(f"\nMode: FULL_DUPLEX")
        self._print(f"  Mic      : [{mic_idx}]")
        self._print(f"  Speaker  : [{spk_idx}]")
        self._print(f"  Ref mode : {'monitor (ref=output)' if use_monitor else f'ref device [{ref_idx}]'}")
        self._print(f"  K (HA)   : {20.0 * np.log10(self.output_gain):.1f} dB (gain={self.output_gain:.2f}x)")
        self._print(f"  AGC      : {'on' if self._agc_enabled else 'off'}")
        self._print(f"  Algorithm: {self.afc.algo_name}")

        try:
            mic_stream = self.pyaudio.open(
                format=pyaudio.paFloat32, channels=1,
                rate=self.sample_rate, input=True,
                input_device_index=mic_idx,
                frames_per_buffer=self.chunk_size,
            )
            out_stream = self.pyaudio.open(
                format=pyaudio.paFloat32, channels=1,
                rate=self.sample_rate, output=True,
                output_device_index=spk_idx,
                frames_per_buffer=self.chunk_size,
            )
            ref_stream = None
            if not use_monitor:
                ref_stream = self.pyaudio.open(
                    format=pyaudio.paFloat32, channels=1,
                    rate=self.sample_rate, input=True,
                    input_device_index=ref_idx,
                    frames_per_buffer=self.chunk_size,
                )
        except Exception as e:
            self._print(f"\nLoi mo stream: {e}")
            return

        self._reset_startup_ramp()
        self.running = True
        self._print(f"\n{'='*60}")
        self._print(f"CHAY - AFC FULL_DUPLEX")
        self._print(f"  SR      : {self.sample_rate} Hz")
        self._print(f"  Chunk   : {self.chunk_size}")
        self._print(f"  Taps M  : {self.afc.M}")
        self._print(f"  Algo    : {self.afc.algo_name}")
        self._print(f"  Ramp    : {self._ramp_duration_sec}s  (speaker muted while AFC learns feedback path)")
        self._print(f"  Step    : {self.afc.mu}")
        self._print(f"  Delta   : {self.afc.delta}")
        self._print(f"  Alpha a : {self.afc.a}")
        self._print(f"  Proj p  : {self.afc.p}")
        self._print(f"  W norm  : {self.afc.w_max_norm}")
        self._print(f"  Ctrl+C  : Dung")
        self._print(f"{'='*60}")

        overflow_count = 0
        last_print = time.time()
        sample_count = 0
        smooth_e = 0.0

        while mic_stream.is_active() and out_stream.is_active():
            try:
                mic_data = mic_stream.read(self.chunk_size, exception_on_overflow=False)
                mic_chunk = np.frombuffer(mic_data, dtype=np.float32)

                if ref_stream:
                    ref_data = ref_stream.read(self.chunk_size, exception_on_overflow=False)
                    ref_chunk = np.frombuffer(ref_data, dtype=np.float32)
                else:
                    # Monitor mode: ref = output chunk from previous iteration
                    # (aligns with acoustic path delay naturally via 1-chunk latency)
                    ref_chunk = self.ref_buffer.copy()

                out_chunk, y_est = self.afc.process_chunk(mic_chunk, ref_chunk)

                # ── Startup ramp: update factor ──
                if not self._ramp_done:
                    self._apply_startup_ramp(time.time())

                # --- Stability detection via error RMS tracking ---
                # Update circular RMS history buffer
                chunk_rms = float(np.sqrt(np.mean(out_chunk ** 2)))
                self._error_rms_history[self._error_rms_head] = chunk_rms
                self._error_rms_head = (self._error_rms_head + 1) % len(self._error_rms_history)
                if self._error_rms_count < len(self._error_rms_history):
                    self._error_rms_count += 1

                # Robust expected RMS: median of history (ignores outliers)
                n = self._error_rms_count
                if n >= 8:
                    buf = self._error_rms_history[:n]
                    self._expected_rms = float(np.median(buf))

                # Detect instability: error RMS >> expected (feedback surging)
                ratio = chunk_rms / (self._expected_rms + 1e-12)
                if not self._stable:
                    # Already unstable: stay at mu2 until RMS drops below hysteresis
                    if ratio < self._stable_hysteresis:
                        self._stable_count += 1
                        if self._stable_count >= self._stable_min_frames:
                            self._stable = True
                            self._stable_count = 0
                            self.afc.switch_mu(stable=True)
                else:
                    # Stable: switch to mu2 only when error surges significantly
                    if ratio > self._stable_threshold:
                        self._stable = False
                        self._stable_count = 0
                        self.afc.switch_mu(stable=False)

                # --- Hearing-aid amplification + gain management ---
                # out = K × error (feedback-cancelled speech).
                # Gain floor is unified across both modes: 0 dB (unity) — see below for rationale.
                out_chunk = out_chunk * self.output_gain

                # Apply startup ramp: keep output at 0 during ramp so AFC can learn.
                # When ramp is done, ramp_factor == 1.0 and this is a no-op.
                if not self._ramp_done:
                    out_chunk = out_chunk * self._ramp_factor

                # GAIN_FLOOR = 1.0 (0 dB). Reason: with W_norm declining,
                # any K > 1/W_norm maintains loop gain > 1. Floor must be 0 dB to let
                # AFC converge before gain ramps back up.
                # GAIN_CEIL  = _AFP.K = 30 dB (MATLAB maximum).
                _GAIN_FLOOR = 1.0
                _GAIN_CEIL  = _AFP.K

                self._clipped_ratio = np.sum(np.abs(out_chunk) > 1.0) / self.chunk_size
                clipped = np.sum(np.abs(out_chunk) > 1.0)
                if clipped > 0:
                    # Hard clip at ±1.0 — no tanh distortion.
                    # TanH soft-clipping was removed because it distorts the reference
                    # signal in monitor mode: the clipped output gets saved to ref_buffer,
                    # corrupting AFC's learning signal and causing W_norm to diverge.
                    out_chunk = np.clip(out_chunk, -1.0, 1.0)
                    # Fast gain reduction when clip > 10%: exit feedback loop quickly
                    if self._agc_enabled and self._clipped_ratio > 0.10:
                        self.output_gain = max(self.output_gain * 0.90, _GAIN_FLOOR)
                else:
                    # Slow recovery, ONLY when clip < 1% (nearly no clipping).
                    # 0.5%/chunk — prevents premature gain increase before filter stabilizes.
                    if self._agc_enabled and self.output_gain < _GAIN_CEIL:
                        self.output_gain = min(self.output_gain * 1.005, _GAIN_CEIL)

                # Update ref buffer to feed AFC's next iteration.
                # 
                # CRITICAL: In monitor mode, we must NOT feed AFC output back into AFC input.
                # Using y_est as reference creates exponential growth:
                #   y_est[n] = gTD × y_est[n-1]  →  y_est[n] = gTD^n × y_est[0]
                # This causes RMS to explode (confirmed: 1.5e-5 → 511 in 10 chunks).
                #
                # Solution: In monitor mode, use ZERO as reference during ramp.
                # This prevents AFC feedback loop while still allowing AFC to process.
                # After ramp completes, we can switch to a proper reference if available.
                #
                # Note: This means AFC won't learn during ramp, but that's acceptable
                # because gain is zero anyway (no speaker output = no feedback to learn).
                if not use_monitor:
                    # ref device mode: hardware reference - use output (this is correct)
                    self.ref_buffer[:] = out_chunk
                else:
                    # monitor mode: break AFC feedback loop by using zero reference
                    # AFC processes the input but doesn't feed its output back
                    self.ref_buffer.fill(0.0)

                out_stream.write(out_chunk.astype(np.float32).tobytes())
                sample_count += self.chunk_size

                # STOI/PESQ: clean = mic (user's speech), processed = output (user hears)
                for i in range(self.chunk_size):
                    self._push_stoi(float(mic_chunk[i]), float(out_chunk[i]))
                    self._push_pesq(float(mic_chunk[i]), float(out_chunk[i]))

                now = time.time()
                if now - last_print >= 2.0:
                    s = self.afc.get_stats()
                    smooth_e = (1 - self._smooth_alpha) * smooth_e + self._smooth_alpha * chunk_rms

                    if now - self._stoi_last_compute >= self._stoi_win_sec:
                        self._current_stoi = self._compute_stoi()
                        self._stoi_last_compute = now

                    if now - self._pesq_last_compute >= self._pesq_win_sec:
                        self._current_pesq, self._current_snrseg, self._current_lsd = self._compute_pesq()
                        self._pesq_last_compute = now

                    # Build display values with type safety + debug for tuple sources
                    # Always safe-convert first, then debug types
                    def _safe(v, default=0.0):
                        try: return float(v) if not isinstance(v, tuple) else default
                        except: return default
                    def _safe_int(v, default=0):
                        try: return int(v) if not isinstance(v, tuple) else default
                        except: return default

                    _t = type(sample_count).__name__
                    if _t == 'tuple':
                        sys.stderr.write(f"DEBUG: sample_count is tuple: {sample_count}\n")
                    _t2 = type(smooth_e).__name__
                    if _t2 == 'tuple':
                        sys.stderr.write(f"DEBUG: smooth_e is tuple: {smooth_e}\n")
                    _t3 = type(s['w_norm']).__name__
                    if _t3 == 'tuple':
                        sys.stderr.write(f"DEBUG: w_norm is tuple: {s['w_norm']}\n")
                    _t4 = type(clipped).__name__
                    if _t4 == 'tuple':
                        sys.stderr.write(f"DEBUG: clipped is tuple: {clipped}\n")
                    _t5 = type(self._clipped_ratio).__name__
                    if _t5 == 'tuple':
                        sys.stderr.write(f"DEBUG: _clipped_ratio is tuple: {self._clipped_ratio}\n")
                    _t6 = type(self.output_gain).__name__
                    if _t6 == 'tuple':
                        sys.stderr.write(f"DEBUG: output_gain is tuple: {self.output_gain}\n")
                    _t7 = type(self._current_stoi).__name__
                    if _t7 == 'tuple':
                        sys.stderr.write(f"DEBUG: _current_stoi is tuple: {self._current_stoi}\n")
                    _t8 = type(self._current_pesq).__name__
                    if _t8 == 'tuple':
                        sys.stderr.write(f"DEBUG: _current_pesq is tuple: {self._current_pesq}\n")

                    disp_sample_count = _safe_int(sample_count)
                    disp_smooth_e = _safe(smooth_e)
                    disp_w_norm = _safe(s['w_norm'])
                    disp_clipped = _safe_int(clipped)
                    disp_clipped_ratio = _safe(self._clipped_ratio)
                    disp_gain_db = _safe(20.0*np.log10(self.output_gain+1e-12))
                    disp_stoi = _safe(self._current_stoi)
                    disp_pesq = _safe(self._current_pesq)
                    disp_ramp_pct = int(self._ramp_factor * 100)
                    disp_ramp_str = f"RAMP:{disp_ramp_pct}%" if not self._ramp_done else "RAMP:DONE"
                    wmax_str = 'inf' if self.afc.w_max_norm == float('inf') else f'{self.afc.w_max_norm:.1f}'
                    sw_str = 'mu1' if self._stable else 'mu2'

                    self._print(
                        f"  [{time.strftime('%H:%M:%S')}] "
                        f"Samples: {disp_sample_count:>10,} | "
                        f"RMS: {disp_smooth_e:.5f} | "
                        f"W norm: {disp_w_norm:.4f}/{wmax_str} | "
                        f"Pre-clip: {disp_clipped}/{self.chunk_size} ({disp_clipped_ratio*100:.0f}%) | "
                        f"K: {disp_gain_db:.1f}dB | "
                        f"SW: {sw_str} | "
                        f"{disp_ramp_str} | "
                        f"STOI: {disp_stoi:.3f} | "
                        f"PESQ: {disp_pesq:.2f}",
                        end='\r'
                    )
                    last_print = now

            except IOError:
                overflow_count += 1
                self._print(f"\n  [Overflow #{overflow_count}]")
                time.sleep(0.01)

        self.running = False
        self._print("\n\nDang dung...")
        mic_stream.stop_stream()
        mic_stream.close()
        out_stream.stop_stream()
        out_stream.close()
        if ref_stream:
            ref_stream.stop_stream()
            ref_stream.close()
        self._print("Da dung.")

    def run_loopback(self, mic_idx, loopback_idx, spk_idx=None):
        self.mode = self.MODE_LOOPBACK
        self._print(f"\nMode: LOOPBACK")
        self._print(f"  Mic      : [{mic_idx}]")
        self._print(f"  Loopback : [{loopback_idx}]")
        self._print(f"  Spk      : [{spk_idx}]")
        self._print(f"  K (HA)   : {20.0 * np.log10(self.output_gain):.1f} dB (gain={self.output_gain:.2f}x)")
        self._print(f"  AGC      : {'on' if self._agc_enabled else 'off'}")
        self._print(f"  Algorithm: {self.afc.algo_name}")

        try:
            mic_stream = self.pyaudio.open(
                format=pyaudio.paFloat32, channels=1,
                rate=self.sample_rate, input=True,
                input_device_index=mic_idx,
                frames_per_buffer=self.chunk_size,
            )
            loop_stream = self.pyaudio.open(
                format=pyaudio.paFloat32, channels=1,
                rate=self.sample_rate, input=True,
                input_device_index=loopback_idx,
                frames_per_buffer=self.chunk_size,
            )
            out_stream = None
            if spk_idx is not None:
                out_stream = self.pyaudio.open(
                    format=pyaudio.paFloat32, channels=1,
                    rate=self.sample_rate, output=True,
                    output_device_index=spk_idx,
                    frames_per_buffer=self.chunk_size,
                )
        except Exception as e:
            self._print(f"\nLoi mo stream: {e}")
            return

        self._reset_startup_ramp()
        self.running = True
        self._print(f"\n{'='*60}")
        self._print(f"CHAY - AFC LOOPBACK")
        self._print(f"  Algo    : {self.afc.algo_name}")
        self._print(f"  Ramp    : {self._ramp_duration_sec}s  (speaker muted while AFC learns feedback path)")
        self._print(f"{'='*60}")

        overflow_count = 0
        last_print = time.time()
        sample_count = 0
        smooth_e = 0.0

        while mic_stream.is_active() and loop_stream.is_active():
            try:
                mic_data = mic_stream.read(self.chunk_size, exception_on_overflow=False)
                mic_chunk = np.frombuffer(mic_data, dtype=np.float32)

                loop_data = loop_stream.read(self.chunk_size, exception_on_overflow=False)
                loop_chunk = np.frombuffer(loop_data, dtype=np.float32)

                # In loopback: mic captures the acoustic output from speaker
                # (the signal that goes back into the hearing aid).
                # loop_chunk = speaker_output (the reference signal to AFC).
                # AFC learns the acoustic path and outputs error = mic - estimate.
                out_chunk, _ = self.afc.process_chunk(mic_chunk, loop_chunk)

                # ── Startup ramp: update factor ──
                if not self._ramp_done:
                    self._apply_startup_ramp(time.time())

                # --- Stability detection via error RMS tracking ---
                chunk_rms = float(np.sqrt(np.mean(out_chunk ** 2)))
                self._error_rms_history[self._error_rms_head] = chunk_rms
                self._error_rms_head = (self._error_rms_head + 1) % len(self._error_rms_history)
                if self._error_rms_count < len(self._error_rms_history):
                    self._error_rms_count += 1

                n = self._error_rms_count
                if n >= 8:
                    buf = self._error_rms_history[:n]
                    self._expected_rms = float(np.median(buf))

                ratio = chunk_rms / (self._expected_rms + 1e-12)
                if not self._stable:
                    if ratio < self._stable_hysteresis:
                        self._stable_count += 1
                        if self._stable_count >= self._stable_min_frames:
                            self._stable = True
                            self._stable_count = 0
                            self.afc.switch_mu(stable=True)
                else:
                    if ratio > self._stable_threshold:
                        self._stable = False
                        self._stable_count = 0
                        self.afc.switch_mu(stable=False)

                # --- Hearing-aid amplification + gain management ---
                # Gain floor unified to 0 dB (1.0) across both modes.
                out_chunk = out_chunk * self.output_gain

                # Apply startup ramp: keep output at 0 during ramp so AFC can learn.
                if not self._ramp_done:
                    out_chunk = out_chunk * self._ramp_factor

                _GAIN_FLOOR = 1.0          # 0 dB — unified with full_duplex
                _GAIN_CEIL  = _AFP.K       # 30 dB

                self._clipped_ratio = np.sum(np.abs(out_chunk) > 1.0) / self.chunk_size
                clipped = np.sum(np.abs(out_chunk) > 1.0)
                if clipped > 0:
                    # Hard clip at ±1.0 — no tanh distortion.
                    # TanH soft-clipping was removed because it distorts the speaker output
                    # that gets read back by the loopback stream, corrupting the AFC reference.
                    out_chunk = np.clip(out_chunk, -1.0, 1.0)
                    if self._agc_enabled and self._clipped_ratio > 0.10:
                        self.output_gain = max(self.output_gain * 0.90, _GAIN_FLOOR)
                else:
                    if self._agc_enabled and self.output_gain < _GAIN_CEIL:
                        self.output_gain = min(self.output_gain * 1.005, _GAIN_CEIL)

                if out_stream:
                    out_stream.write(out_chunk.astype(np.float32).tobytes())

                sample_count += self.chunk_size

                # STOI/PESQ: clean = mic (user's speech), processed = output (user hears)
                for i in range(self.chunk_size):
                    self._push_stoi(float(mic_chunk[i]), float(out_chunk[i]))
                    self._push_pesq(float(mic_chunk[i]), float(out_chunk[i]))

                now = time.time()
                if now - last_print >= 2.0:
                    s = self.afc.get_stats()
                    smooth_e = (1 - self._smooth_alpha) * smooth_e + self._smooth_alpha * chunk_rms

                    if now - self._stoi_last_compute >= self._stoi_win_sec:
                        self._current_stoi = self._compute_stoi()
                        self._stoi_last_compute = now

                    if now - self._pesq_last_compute >= self._pesq_win_sec:
                        self._current_pesq, self._current_snrseg, self._current_lsd = self._compute_pesq()
                        self._pesq_last_compute = now

                    def _safe(v, default=0.0):
                        try: return float(v) if not isinstance(v, tuple) else default
                        except: return default
                    def _safe_int(v, default=0):
                        try: return int(v) if not isinstance(v, tuple) else default
                        except: return default

                    _t = type(sample_count).__name__
                    if _t == 'tuple':
                        sys.stderr.write(f"DEBUG: sample_count is tuple: {sample_count}\n")
                    _t2 = type(smooth_e).__name__
                    if _t2 == 'tuple':
                        sys.stderr.write(f"DEBUG: smooth_e is tuple: {smooth_e}\n")
                    _t3 = type(s['w_norm']).__name__
                    if _t3 == 'tuple':
                        sys.stderr.write(f"DEBUG: w_norm is tuple: {s['w_norm']}\n")
                    _t4 = type(clipped).__name__
                    if _t4 == 'tuple':
                        sys.stderr.write(f"DEBUG: clipped is tuple: {clipped}\n")
                    _t5 = type(self._clipped_ratio).__name__
                    if _t5 == 'tuple':
                        sys.stderr.write(f"DEBUG: _clipped_ratio is tuple: {self._clipped_ratio}\n")
                    _t6 = type(self.output_gain).__name__
                    if _t6 == 'tuple':
                        sys.stderr.write(f"DEBUG: output_gain is tuple: {self.output_gain}\n")
                    _t7 = type(self._current_stoi).__name__
                    if _t7 == 'tuple':
                        sys.stderr.write(f"DEBUG: _current_stoi is tuple: {self._current_stoi}\n")
                    _t8 = type(self._current_pesq).__name__
                    if _t8 == 'tuple':
                        sys.stderr.write(f"DEBUG: _current_pesq is tuple: {self._current_pesq}\n")
                    _t9 = type(self._current_snrseg).__name__
                    if _t9 == 'tuple':
                        sys.stderr.write(f"DEBUG: _current_snrseg is tuple: {self._current_snrseg}\n")
                    _t10 = type(self._current_lsd).__name__
                    if _t10 == 'tuple':
                        sys.stderr.write(f"DEBUG: _current_lsd is tuple: {self._current_lsd}\n")

                    disp_sample_count = _safe_int(sample_count)
                    disp_smooth_e = _safe(smooth_e)
                    disp_w_norm = _safe(s['w_norm'])
                    disp_clipped = _safe_int(clipped)
                    disp_clipped_ratio = _safe(self._clipped_ratio)
                    disp_gain_db = _safe(20.0*np.log10(self.output_gain+1e-12))
                    disp_stoi = _safe(self._current_stoi)
                    disp_pesq = _safe(self._current_pesq)
                    disp_snrseg = _safe(self._current_snrseg)
                    disp_lsd = _safe(self._current_lsd)
                    disp_ramp_pct = int(self._ramp_factor * 100)
                    disp_ramp_str = f"RAMP:{disp_ramp_pct}%" if not self._ramp_done else "RAMP:DONE"
                    wmax_str = 'inf' if self.afc.w_max_norm == float('inf') else f'{self.afc.w_max_norm:.1f}'
                    sw_str = 'mu1' if self._stable else 'mu2'

                    self._print(
                        f"  [{time.strftime('%H:%M:%S')}] "
                        f"Samples: {disp_sample_count:>10,} | "
                        f"RMS: {disp_smooth_e:.5f} | "
                        f"W norm: {disp_w_norm:.4f}/{wmax_str} | "
                        f"Pre-clip: {disp_clipped}/{self.chunk_size} ({disp_clipped_ratio*100:.0f}%) | "
                        f"K: {disp_gain_db:.1f}dB | "
                        f"SW: {sw_str} | "
                        f"{disp_ramp_str} | "
                        f"STOI: {disp_stoi:.3f} | "
                        f"PESQ: {disp_pesq:.2f} | "
                        f"SNRseg: {disp_snrseg:.1f}dB | "
                        f"LSD: {disp_lsd:.2f}dB",
                        end='\r'
                    )
                    last_print = now

            except IOError:
                overflow_count += 1
                time.sleep(0.01)

        self.running = False
        self._print("\n\nDang dung...")
        mic_stream.stop_stream()
        mic_stream.close()
        loop_stream.stop_stream()
        loop_stream.close()
        if out_stream:
            out_stream.stop_stream()
            out_stream.close()
        self._print("Da dung.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def check_dependencies():
    missing = []
    for lib in ['numpy', 'pyaudio']:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"Thieu: {', '.join(missing)} -> pip install {' '.join(missing)}")
        sys.exit(1)


def auto_tune(sample_rate, algo='nlms'):
    """Tu dong chon thong so toi uu theo MATLAB/Simulink reference.

    For AFC at 16 kHz: M=64, p=2, delta=1e-6, a=0.5.
    mu is algorithm-dependent:
      - nlms/ipnlms  : 1e-3
      - apa/ipapa    : 9e-4
      - ipapsa family: 8e-6
    """
    from adaptfilt.afc_params import AFCParameters as _AFP

    algo_defaults = {
        'nlms':      _AFP.mu_nlms_ipnlms,
        'ipnlms':    _AFP.mu_nlms_ipnlms,
        'apa':       _AFP.mu_apa_ipapa,
        'ipapa':     _AFP.mu_apa_ipapa,
        'ipapsa':    _AFP.mu_ipapsa_family,
        'mipapsa':   _AFP.mu_ipapsa_family,
        'bsmipapsa': _AFP.mu_ipapsa_family,
    }
    step = algo_defaults.get(algo, _AFP.mu_nlms_ipnlms)

    if sample_rate <= 16000:
        chunk = 256
    else:
        chunk = 512
    M, delta, a, p = 64, 1e-6, 0.5, 2
    return M, step, delta, a, p, chunk


def main():
    check_dependencies()

    # ---- INTERACTIVE MENU ----
    print("\n" + "=" * 60)
    print("   AFC REALTIME - 7 THUAT TOAN THICH NGHI")
    print("=" * 60)
    print()
    print("  [1] NLMS       - Normalized LMS (don gian, on dinh)")
    print("  [2] IPNLMS     - Improved P-NLMS (phan chia thich nghi)")
    print("  [3] APA        - Affine Projection Algo")
    print("  [4] IPAPA      - Improved P-APA")
    print("  [5] IPAPSA     - Improved P-APSA (signed error)")
    print("  [6] MIPAPSA    - Memory IP-APSA")
    print("  [7] BSMIPAPSA  - Block-sparse MIPAPSA")
    print("  [A] Tat ca      - Chay tat ca thuat toan lien tiep")
    print()
    choice_map = {
        '1': 'nlms', '2': 'ipnlms', '3': 'apa', '4': 'ipapa',
        '5': 'ipapsa', '6': 'mipapsa', '7': 'bsmipapsa', 'a': 'all'
    }
    algo_map = {
        'nlms': 'PEM-NLMS', 'ipnlms': 'PEM-IPNLMS', 'apa': 'PEM-APA',
        'ipapa': 'PEM-IPAPA', 'ipapsa': 'PEM-IPAPSA',
        'mipapsa': 'PEM-MIPAPSA', 'bsmipapsa': 'PEM-BSMIPAPSA'
    }
    while True:
        sel = input("  Chon thuat toan [1-7, A=all]: ").strip().lower()
        if sel in choice_map:
            break
        print("  [!] Vui long nhap 1-7 hoac A")
    selected_algo = choice_map[sel]

    print()
    print("=" * 60)
    print("AFC REALTIME - 7 THUAT TOAN THICH NGHI")
    print("=" * 60)

    p = pyaudio.PyAudio()
    try:
        default_in = p.get_default_input_device_info()
        default_out = p.get_default_output_device_info()
        mic_idx = default_in['index']
        spk_idx = default_out['index']
        sr = int(default_in.get('defaultSampleRate', 44100))
    except Exception:
        sr = 44100
        mic_idx = 0
        spk_idx = 2
    p.terminate()

    import argparse
    parser = argparse.ArgumentParser(
        description='AFC v5 - 7 Thuat toan: NLMS, IPNLMS, APA, IPAPA, IPAPSA, MIPAPSA, BSMIPAPSA'
    )
    parser.add_argument('--algo', type=str, default='nlms',
                        choices=list(AdaptiveFilter.ALGORITHMS.keys()),
                        help='Thu tu toan thich nghi (default: nlms)')
    parser.add_argument('--mic', type=int, default=mic_idx)
    parser.add_argument('--spk', type=int, default=spk_idx)
    parser.add_argument('--ref', type=int, default=None)
    parser.add_argument('--loopback', action='store_true')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate in Hz (default: 16000, MATLAB reference)')
    parser.add_argument('--chunk', type=int, default=None)
    parser.add_argument('--M', type=int, default=None,
                        help='Filter order (default: 64, MATLAB reference)')
    parser.add_argument('--step', type=float, default=None,
                        help='Step size / mu (default: algorithm-dependent)')
    parser.add_argument('--delta', type=float, default=None,
                        help='Regularization (default: 1e-6)')
    parser.add_argument('--a', type=float, default=None,
                        help='Alpha for proportionate algorithms (default: 0.5)')
    parser.add_argument('--p', type=int, default=None,
                        help='Projection order for APA family (default: 2, MATLAB reference)')
    parser.add_argument('--wmax', type=float, default=20.0,
                        help='Max weight norm (default: 20.0, tranh blow-up bo loc)')
    parser.add_argument('--leaky', type=float, default=None,
                        help='Leakage factor (default: 0.0)')
    parser.add_argument('--gain', type=float, default=None,
                        help='Linear gain (default: 25 dB startup, 30 dB ceiling)')
    parser.add_argument('--db', type=float, default=30.0,
                        help='Amplification in dB (default: 30, MATLAB Kdb)')
    parser.add_argument('--auto', action='store_true',
                        help='Tu dong chon thong so')
    # --- New MATLAB/Simulink parameters ---
    parser.add_argument('--mu1', type=float, default=None,
                        help='NLMS step when stable (default: 4e-6, MATLAB)')
    parser.add_argument('--mu2', type=float, default=None,
                        help='NLMS step when unstable (default: 8e-2, MATLAB)')
    parser.add_argument('--eps', type=float, default=None,
                        help='Regression matrix epsilon (default: 1e-5, MATLAB)')
    parser.add_argument('--lda', type=float, default=None,
                        help='Signed error sigmoid coefficient (default: 6, MATLAB)')
    parser.add_argument('--delta-sc', type=float, default=None,
                        dest='delta_sc',
                        help='Sparseness measure regularization (default: 1e-8, MATLAB)')
    parser.add_argument('--beta', type=float, default=None,
                        help='TanH-gated threshold factor (default: 50, MATLAB)')
    parser.add_argument('--Mbs', type=int, default=None,
                        dest='M_bs',
                        help='Number of blocks for BSMIPAPSA (default: 8, MATLAB)')
    parser.add_argument('--tanh-scale', type=float, default=None,
                        dest='tanh_scale',
                        help='Error clipping scale (default: 0.5, MATLAB)')
    parser.add_argument('--tanh-thresh', type=float, default=None,
                        dest='tanh_thresh',
                        help='Impulse detection threshold (default: 0.15, MATLAB)')
    parser.add_argument('--max-ar', type=float, default=None,
                        dest='max_ar',
                        help='AR coefficient clamp limit (default: 6, MATLAB)')
    parser.add_argument('--dk', type=int, default=None,
                        help='Forward path delay samples (default: 96, MATLAB)')
    parser.add_argument('--dfb', type=int, default=None,
                        help='Feedback path delay samples (default: 1, MATLAB)')
    parser.add_argument('--stoi-win', type=float, default=1.0,
                        help='STOI computation window in seconds (default: 1.0)')
    parser.add_argument('--pesq-win', type=float, default=4.0,
                        help='PESQ computation window in seconds (default: 4.0, needs ~1s+ signal)')
    # Khi chay tu menu, luon dung selected_algo; override args.algo
    args = parser.parse_args()
    args.algo = selected_algo
    args.loopback = False  # menu mac dinh monitor mode

    mode = 'loopback' if args.loopback else 'monitor'
    # Gain khởi tạo 10dB (3.162x) thay vì 30dB
    # Lý do: gain cao gây feedback loop ngay lập tức, filter chưa kịp học
    # 10dB đủ để nghe rõ trong khi filter thích nghi
    # Người dùng có thể override bằng --gain hoặc --db
    # Startup default gain: 25 dB (gain starts below the 30 dB ceiling to give
    # the AFC filter room to learn before reaching full amplification).
    _DEFAULT_GAIN_DB = 25.0
    _DEFAULT_LINEAR_GAIN = 10 ** (_DEFAULT_GAIN_DB / 20.0)  # ~17.78x
    linear_gain = args.gain if args.gain is not None else _DEFAULT_LINEAR_GAIN
    db_gain = 20.0 * np.log10(linear_gain)

    # Auto-tune (base values — step will be overridden per-algo inside the loop below)
    if args.auto:
        M, _step_auto, delta, a, p, chunk = auto_tune(args.sr, algo=selected_algo)
        if args.M is not None:
            M = args.M
        if args.delta is not None:
            delta = args.delta
        if args.a is not None:
            a = args.a
        if args.p is not None:
            p = args.p
        if args.chunk is not None:
            chunk = args.chunk
    else:
        chunk = args.chunk if args.chunk else 512
        M = args.M if args.M else 64
        delta = args.delta if args.delta else 1e-6
        a = args.a if args.a else 0.5
        p = args.p if args.p else 2
        # step is determined per algorithm below, after algo_key is known

    # List devices
    p2 = pyaudio.PyAudio()
    print(f"\nThiet bi (SR: {args.sr} Hz, MATLAB reference fs=16000 Hz):")
    print("-" * 60)
    info = p2.get_host_api_info_by_index(0)
    device_list = []
    for i in range(info.get('deviceCount', 0)):
        try:
            d = p2.get_device_info_by_host_api_device_index(0, i)
            n = d.get('name', '')
            in_ch = d.get('maxInputChannels', 0)
            out_ch = d.get('maxOutputChannels', 0)
            if in_ch > 0 or out_ch > 0:
                tags = []
                if in_ch > 0:
                    tags.append(f"IN:{in_ch}")
                if out_ch > 0:
                    tags.append(f"OUT:{out_ch}")
                print(f"  [{i}] {n:<45} {' | '.join(tags)}")
                device_list.append((i, n, in_ch, out_ch))
        except Exception:
            pass
    p2.terminate()
    print("-" * 60)

    # Chon thiet bi
    print(f"\nMic  : [{args.mic}]")
    print(f"Loa  : [{args.spk}]")
    print(f"Ref  : {'monitor' if args.ref is None else f'[{args.ref}]'}")
    print(f"Mode : {mode.upper()}")
    print(f"\nThu tu toan: {', '.join(AdaptiveFilter.ALGORITHMS.keys())}")

    # Danh sach thuat toan can chay
    if selected_algo == 'all':
        algo_list = list(AdaptiveFilter.ALGORITHMS.keys())
    else:
        algo_list = [selected_algo]

    for idx, algo_key in enumerate(algo_list):
        print()
        print("=" * 60)
        print(f"  [{idx+1}/{len(algo_list)}] CHAY THUAT TOAN: {algo_key.upper()}")
        print("=" * 60)

        algo_display = AdaptiveFilter.ALGORITHMS.get(algo_key, algo_key.upper())

        # step per-algorithm:
        #   - CLI --step → use it directly
        #   - --auto     → call auto_tune with current algo_key for correct per-algo mu
        #   - neither    → let AdaptiveFilter pick per-algo default from AFCParameters
        if args.step is not None:
            step = args.step
        elif args.auto:
            _, step, _, _, _, _ = auto_tune(args.sr, algo=algo_key)
            if args.chunk is not None:
                chunk = args.chunk
        else:
            step = None
        step_display = step if step is not None else f"auto ({_AFP.get_mu(algo_key)})"

        print(f"\nThong so AFC:")
        print(f"  Algo      : {algo_display}")
        print(f"  M         : {M}")
        print(f"  Step      : {step_display}")
        print(f"  Delta     : {delta}")
        print(f"  Alpha a   : {a}")
        print(f"  Proj p    : {p}")
        print(f"  W max     : {args.wmax if args.wmax else 'inf (khong gioi han)'}")
        print(f"  Leaky     : {args.leaky if args.leaky is not None else 0.0}")
        print(f"  Chunk     : {chunk}")
        print(f"  SR        : {args.sr} Hz")
        print(f"  Gain      : {db_gain:.1f} dB ({linear_gain:.2f}x)")
        print(f"  -- AFC MATLAB params --")
        print(f"  mu1       : {args.mu1 if args.mu1 is not None else 4e-6}")
        print(f"  mu2       : {args.mu2 if args.mu2 is not None else 8e-2}")
        print(f"  eps       : {args.eps if args.eps is not None else 1e-5}")
        print(f"  lda       : {args.lda if args.lda is not None else 6}")
        print(f"  beta      : {args.beta if args.beta is not None else 50}")
        print(f"  delta_sc  : {args.delta_sc if args.delta_sc is not None else 1e-8}")
        print(f"  M_bs      : {args.M_bs if args.M_bs is not None else 8}")
        print(f"  tanh_scale: {args.tanh_scale if args.tanh_scale is not None else 0.5}")
        print(f"  tanh_thres: {args.tanh_thresh if args.tanh_thresh is not None else 0.15}")
        print(f"  max_ar    : {int(args.max_ar) if args.max_ar is not None else 6}")
        print(f"  d_k       : {args.dk if args.dk is not None else 96}")
        print(f"  d_fb      : {args.dfb if args.dfb is not None else 1}")

        afc = AdaptiveFilter(
            M=M,
            step=step,
            delta=delta,
            a=a,
            p=p,
            algo=algo_key,
            w_max_norm=args.wmax,
            leaky=args.leaky if args.leaky is not None else 0.0,
            mu1=args.mu1 if args.mu1 is not None else 4e-6,
            mu2=args.mu2 if args.mu2 is not None else 8e-2,
            eps=args.eps if args.eps is not None else 1e-5,
            lda=args.lda if args.lda is not None else 6,
            delta_sc=args.delta_sc if args.delta_sc is not None else 1e-8,
            beta=args.beta if args.beta is not None else 50,
            M_bs=args.M_bs if args.M_bs is not None else 8,
            tanh_scale=args.tanh_scale if args.tanh_scale is not None else 0.5,
            tanh_thresh=args.tanh_thresh if args.tanh_thresh is not None else 0.15,
            max_ar=int(args.max_ar) if args.max_ar is not None else 6,
            fs=args.sr,
            d_k=args.dk if args.dk is not None else 96,
            d_fb=args.dfb if args.dfb is not None else 1,
        )

        manager = AFCManager(afc=afc, sample_rate=args.sr, chunk_size=chunk,
                             stoi_win_sec=args.stoi_win,
                             pesq_win_sec=args.pesq_win)
        manager.output_gain = linear_gain

        if args.loopback:
            loopback_idx = args.ref if args.ref else args.spk
            manager.run_loopback(args.mic, loopback_idx,
                                 None if args.ref is None else args.spk)
        else:
            manager.run_full_duplex(args.mic, args.spk, ref_idx=args.ref)

        # Neu chay nhieu thuat toan, cho 3s giua cac lan
        if len(algo_list) > 1 and idx < len(algo_list) - 1:
            print("\n  --> Chuyen sang thuat toan tiep theo trong 3s...")
            time.sleep(3)


if __name__ == '__main__':
    main()