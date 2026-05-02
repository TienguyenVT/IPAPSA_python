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

    ref_max = np.max(np.abs(ref)) + 1e-12
    ref_p = ref / ref_max
    deg_p = deg / ref_max

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

    # Zwicker loudness per band: L_j = S_j^0.3
    ref_bark_loud = np.power(ref_bark, 0.3)
    deg_bark_loud = np.power(deg_bark, 0.3)

    # Asymmetric disturbance per Bark band (P.862 eq. 1-2):
    #   D_plus  = max(deg - ref, 0)  — added noise, penalize fully
    #   D_minus = min(deg - ref, 0)  — missing signal, penalize with alpha=0.5
    # Computing per-band BEFORE summing preserves frequency resolution.
    alpha = 0.5
    diff_band = deg_bark_loud - ref_bark_loud
    D_plus  = np.maximum(diff_band, 0.0)
    D_minus = np.minimum(diff_band, 0.0)
    D_asym_per_band = D_plus + alpha * D_minus   # shape: (n_frames, n_bark)

    # Weight each band by its relative contribution to total reference loudness
    ref_total_loud = np.sum(ref_bark_loud, axis=1, keepdims=True) + 1e-12
    band_weights = ref_bark_loud / ref_total_loud
    D_asym_weighted = D_asym_per_band * band_weights

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


from adaptive_algorithms import AdaptiveFilter

# ------------------------------------------------------------------
# AFCManager - Audio I/O
# ------------------------------------------------------------------
class AFCManager:
    MODE_FULL_DUPLEX = 'full_duplex'
    MODE_LOOPBACK = 'loopback'

    def __init__(self, afc, sample_rate=44100, chunk_size=512,
                 stoi_win_sec=3.0, pesq_win_sec=4.0,
                 stable_threshold=None, stable_hysteresis=None,
                 stable_min_frames=None,
                 min_gain_db=-6.0,
                 log_csv=False, log_wav=False,
                 device_rate=None):
        self.afc = afc
        self.sample_rate = sample_rate   # processing rate (16000 Hz, MATLAB ref)
        self.chunk_size = chunk_size     # chunk size at processing rate

        # --- Sample-rate bridging (FIX: capture at native device rate) ---
        # device_rate: hardware sample rate (e.g. 44100 Hz for Realtek)
        # proc_rate  : AFC processing rate (16000 Hz, MATLAB reference)
        # PyAudio streams are opened at device_rate; data is resampled in Python
        # using high-quality resample_poly before AFC processing and after.
        # This eliminates the low-quality Windows driver SRC that causes glitches.
        self._device_rate = device_rate if device_rate is not None else sample_rate
        self._proc_rate = sample_rate
        _need_resample = (self._device_rate != self._proc_rate)
        self._need_resample = _need_resample
        if _need_resample:
            from math import gcd as _gcd
            _g = _gcd(self._device_rate, self._proc_rate)
            self._rs_up   = self._proc_rate   // _g   # up-factor   (e.g. 160 for 44100→16000)
            self._rs_down = self._device_rate // _g   # down-factor (e.g. 441 for 44100→16000)
            # Device chunk size: number of samples to request from hardware per AFC chunk
            # Must be integer; slight rounding is fine — resample_poly handles fractional ratios.
            self._device_chunk_size = max(1, round(chunk_size * self._device_rate / self._proc_rate))
        else:
            self._rs_up = self._rs_down = 1
            self._device_chunk_size = chunk_size
        self.pyaudio = pyaudio.PyAudio()
        self.running = False
        self.mode = None
        self._log_csv = log_csv
        self._log_wav = log_wav
        self._mic_log = []
        self._spk_log = []

        self.mic_buffer = np.zeros(chunk_size, dtype=np.float32)
        self.ref_buffer = np.zeros(chunk_size, dtype=np.float32)
        self.output_gain = 1.0
        self._smooth_alpha = 0.1
        self._clipped_ratio = 0.0
        # FIX-B4: allow gain to go below 0 dB so AGC can suppress bursts
        # that overwhelm the filter before it converges.  Default −6 dB.
        self._min_gain = 10 ** (min_gain_db / 20.0)

        self._agc_enabled = True

        self._ramp_duration_sec = 5.0
        self._ramp_start_time = None
        self._ramp_done = False
        self._ramp_factor = 0.0

        # Stability detection — thresholds exposed so users can tune without
        # editing code.  Defaults match MATLAB reference values.
        self._stable = True
        self._stable_count = 0
        self._stable_threshold = (stable_threshold if stable_threshold is not None
                                  else _AFP.stable_threshold)
        self._stable_hysteresis = (stable_hysteresis if stable_hysteresis is not None
                                   else _AFP.stable_hysteresis)
        self._stable_min_frames = (stable_min_frames if stable_min_frames is not None
                                  else _AFP.stable_min_frames)
        self._error_rms_history = np.zeros(64, dtype=np.float64)
        self._error_rms_head = 0
        self._error_rms_count = 0
        self._expected_rms = 0.01

        self.stats = {
            'samples_processed': 0,
            'overflow_count': 0,
        }
        self._lock = threading.Lock()
        self._print_lock = threading.Lock()

        # Forward-path delay d_k: delay between AFC error output and gain.
        # This decorrelates near-end speech from feedback — CRITICAL for PEM-AFC.
        # Matches C/Simulink dk_DSTATE[96] block.
        self._dk = _AFP.d_k  # = 96 samples
        self._dk_buf = np.zeros(self._dk, dtype=np.float64)
        self._dk_head = 0  # write pointer into circular buffer

        # Divergence recovery: if AGC is stuck at floor for too long, reset filter
        self._agc_floor_frames = 0
        self._agc_floor_reset_threshold = 200  # ~200 chunks ≈ 6.4s at 512/16kHz

        # STOI buffers
        self._stoi_buffer_len = int(stoi_win_sec * sample_rate)
        self._stoi_clean = np.zeros(self._stoi_buffer_len, dtype=np.float64)
        self._stoi_processed = np.zeros(self._stoi_buffer_len, dtype=np.float64)
        self._stoi_clean_head = 0
        self._stoi_ready = False
        self._stoi_fill_count = 0
        self._stoi_win_sec = stoi_win_sec
        self._stoi_last_compute = 0.0
        self._current_stoi = 0.0

        # PESQ buffers (longer window: ~4s)
        self._pesq_buffer_len = int(pesq_win_sec * sample_rate)
        self._pesq_clean = np.zeros(self._pesq_buffer_len, dtype=np.float64)
        self._pesq_processed = np.zeros(self._pesq_buffer_len, dtype=np.float64)
        self._pesq_clean_head = 0
        self._pesq_ready = False
        self._pesq_fill_count = 0   # FIX-B6: fill counter mirrors STOI approach
        self._pesq_win_sec = pesq_win_sec
        self._pesq_last_compute = 0.0
        self._current_pesq = 0.0
        self._current_snrseg = 0.0
        self._current_lsd = 0.0

        # FIX-PERF: STOI/PESQ chạy trong background thread để KHÔNG block audio loop.
        # Audio loop chỉ copy chunk vào circular buffer (vectorized numpy slice, ~0.003ms).
        # Background thread đọc snapshot định kỳ → compute → ghi vào _current_* qua lock.
        # Kết quả: audio loop không bao giờ bị delay vì STOI/PESQ nặng nề.
        self._metrics_lock = threading.Lock()
        self._metrics_thread = None
        self._metrics_stop = threading.Event()

    def _resample_to_proc(self, data: np.ndarray) -> np.ndarray:
        """Resample from device_rate → proc_rate (e.g. 44100 → 16000 Hz).
        Returns float32 array of length ≈ chunk_size."""
        if not self._need_resample:
            return data
        out = resample_poly(data.astype(np.float64), self._rs_up, self._rs_down)
        # Trim/pad to exact chunk_size to keep downstream buffers stable
        if len(out) > self.chunk_size:
            out = out[:self.chunk_size]
        elif len(out) < self.chunk_size:
            out = np.concatenate([out, np.zeros(self.chunk_size - len(out))])
        return out.astype(np.float32)

    def _resample_to_device(self, data: np.ndarray) -> np.ndarray:
        """Resample from proc_rate → device_rate (e.g. 16000 → 44100 Hz).
        Returns float32 array of length ≈ _device_chunk_size."""
        if not self._need_resample:
            return data
        out = resample_poly(data.astype(np.float64), self._rs_down, self._rs_up)
        # Trim/pad to exact device_chunk_size
        if len(out) > self._device_chunk_size:
            out = out[:self._device_chunk_size]
        elif len(out) < self._device_chunk_size:
            out = np.concatenate([out, np.zeros(self._device_chunk_size - len(out))])
        return out.astype(np.float32)

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

    def _push_metrics_chunk(self, mic_chunk: np.ndarray, err_chunk: np.ndarray):
        """
        FIX-PERF: Ghi cả chunk vào circular buffer bằng numpy slice (không dùng vòng lặp).
        Tốc độ: ~0.003ms/chunk thay vì ~0.4ms/chunk của vòng lặp per-sample gốc.

        Thay thế hoàn toàn _push_stoi() và _push_pesq() per-sample.
        Cả STOI và PESQ dùng chung buffer: stoi_clean/stoi_processed.
        """
        n = len(mic_chunk)
        mic_f64 = mic_chunk.astype(np.float64)
        err_f64 = err_chunk.astype(np.float64)

        # --- STOI circular buffer ---
        h = self._stoi_clean_head
        bl = self._stoi_buffer_len
        end = h + n
        if end <= bl:
            self._stoi_clean[h:end] = mic_f64
            self._stoi_processed[h:end] = err_f64
            self._stoi_clean_head = end % bl
        else:
            split = bl - h
            self._stoi_clean[h:] = mic_f64[:split]
            self._stoi_clean[:n - split] = mic_f64[split:]
            self._stoi_processed[h:] = err_f64[:split]
            self._stoi_processed[:n - split] = err_f64[split:]
            self._stoi_clean_head = n - split
        if not self._stoi_ready:
            self._stoi_fill_count += n
            if self._stoi_fill_count >= bl:
                self._stoi_ready = True

        # --- PESQ circular buffer ---
        h = self._pesq_clean_head
        bl = self._pesq_buffer_len
        end = h + n
        if end <= bl:
            self._pesq_clean[h:end] = mic_f64
            self._pesq_processed[h:end] = err_f64
            self._pesq_clean_head = end % bl
        else:
            split = bl - h
            self._pesq_clean[h:] = mic_f64[:split]
            self._pesq_clean[:n - split] = mic_f64[split:]
            self._pesq_processed[h:] = err_f64[:split]
            self._pesq_processed[:n - split] = err_f64[split:]
            self._pesq_clean_head = n - split
        if not self._pesq_ready:
            self._pesq_fill_count += n
            if self._pesq_fill_count >= bl:
                self._pesq_ready = True

    def _push_stoi(self, clean_sample, processed_sample):
        """Append sample to circular STOI buffers."""
        buf_len = self._stoi_buffer_len
        self._stoi_clean[self._stoi_clean_head] = clean_sample
        self._stoi_processed[self._stoi_clean_head] = processed_sample
        self._stoi_clean_head = (self._stoi_clean_head + 1) % buf_len
        if not self._stoi_ready:
            self._stoi_fill_count += 1
            if self._stoi_fill_count >= buf_len:
                self._stoi_ready = True

    def _metrics_worker(self):
        """
        FIX-PERF: Background thread tính STOI/PESQ/SNRseg/LSD.

        Chạy trong thread riêng để KHÔNG block audio loop.
        Mỗi `interval` giây: lấy snapshot buffer → compute → cập nhật _current_*.
        Audio loop đọc _current_* qua _metrics_lock (non-blocking trylock).
        """
        stoi_interval = max(self._stoi_win_sec, 2.0)
        pesq_interval = max(self._pesq_win_sec, 3.0)
        last_stoi = 0.0
        last_pesq = 0.0

        while not self._metrics_stop.is_set():
            now = time.time()

            # --- STOI ---
            if now - last_stoi >= stoi_interval and self._stoi_ready:
                try:
                    # Snapshot buffer (non-destructive read)
                    with self._metrics_lock:
                        head = self._stoi_clean_head
                        if head == 0:
                            clean_seg = self._stoi_clean.copy()
                            proc_seg = self._stoi_processed.copy()
                        else:
                            clean_seg = np.concatenate([self._stoi_clean[head:],
                                                        self._stoi_clean[:head]])
                            proc_seg = np.concatenate([self._stoi_processed[head:],
                                                       self._stoi_processed[:head]])
                    stoi_val = pystoi.stoi(clean_seg, proc_seg,
                                           self.sample_rate, extended=False)
                    with self._metrics_lock:
                        self._current_stoi = float(stoi_val)
                    last_stoi = now
                except Exception:
                    last_stoi = now  # don't retry immediately on error

            # --- PESQ + SNRseg + LSD ---
            if now - last_pesq >= pesq_interval and self._pesq_ready:
                try:
                    with self._metrics_lock:
                        head = self._pesq_clean_head
                        if head == 0:
                            clean_seg = self._pesq_clean.copy()
                            proc_seg = self._pesq_processed.copy()
                        else:
                            clean_seg = np.concatenate([self._pesq_clean[head:],
                                                        self._pesq_clean[:head]])
                            proc_seg = np.concatenate([self._pesq_processed[head:],
                                                       self._pesq_processed[:head]])
                    if _PESQ_OFFICIAL:
                        mode = 'wb' if self.sample_rate == 16000 else 'nb'
                        res = _pesq_official(self.sample_rate,
                                            clean_seg.astype(np.float32),
                                            proc_seg.astype(np.float32), mode)
                        try:
                            pesq_val = float(getattr(res, 'pesq',
                                            res[0] if hasattr(res, '__getitem__') else res))
                        except Exception:
                            pesq_val = float(res)
                    else:
                        pesq_val = compute_pesq(clean_seg, proc_seg, self.sample_rate)
                    snrseg_val = compute_snr_seg(clean_seg, proc_seg, self.sample_rate)
                    lsd_val = compute_lsd(clean_seg, proc_seg, self.sample_rate)
                    with self._metrics_lock:
                        self._current_pesq = float(pesq_val)
                        self._current_snrseg = float(snrseg_val)
                        self._current_lsd = float(lsd_val)
                    last_pesq = now
                except Exception:
                    last_pesq = now

            # Ngủ ngắn để không spin-loop, nhưng đủ responsive
            self._metrics_stop.wait(timeout=0.5)

    def _start_metrics_thread(self):
        """Khởi động background metrics thread."""
        self._metrics_stop.clear()
        self._metrics_thread = threading.Thread(
            target=self._metrics_worker, daemon=True, name='AFC-Metrics')
        self._metrics_thread.start()

    def _stop_metrics_thread(self):
        """Dừng background metrics thread."""
        self._metrics_stop.set()
        if self._metrics_thread is not None:
            self._metrics_thread.join(timeout=2.0)
            self._metrics_thread = None

    def _compute_stoi(self):
        """Compute STOI from circular buffers. Returns (stoi_val, n_frames)."""
        if not self._stoi_ready:
            return 0.0, 0
        buf_len = self._stoi_buffer_len
        head = self._stoi_clean_head
        if head == 0:
            clean_seg = self._stoi_clean
            proc_seg = self._stoi_processed
        else:
            # Slicing avoids np.roll() full-copy overhead.
            clean_seg = np.concatenate([self._stoi_clean[head:], self._stoi_clean[:head]])
            proc_seg = np.concatenate([self._stoi_processed[head:], self._stoi_processed[:head]])
        try:
            stoi_val = pystoi.stoi(clean_seg, proc_seg, self.sample_rate, extended=False)
        except Exception:
            stoi_val = 0.0
        return float(stoi_val), 0

    def _push_pesq(self, clean_sample, processed_sample):
        """Append sample to circular PESQ buffers.
        NOTE: Dùng _push_metrics_chunk() cho chunk-level writes (nhanh hơn 100x).
        Hàm này giữ lại cho compatibility nhưng không nên gọi trong hot path.
        """
        buf_len = self._pesq_buffer_len
        self._pesq_clean[self._pesq_clean_head] = clean_sample
        self._pesq_processed[self._pesq_clean_head] = processed_sample
        self._pesq_clean_head = (self._pesq_clean_head + 1) % buf_len
        # FIX-PERF: Đã XÓA debug JSON file I/O (open/write) khỏi hot path.
        # Trước đây mỗi buffer wrap gây ra file I/O blocking + json.dumps
        # ngay trong vòng lặp per-sample → unpredictable latency spikes.
        if not self._pesq_ready:
            self._pesq_fill_count += 1
            if self._pesq_fill_count >= buf_len:
                self._pesq_ready = True
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
        head = self._pesq_clean_head
        if head == 0:
            clean_seg = self._pesq_clean
            proc_seg = self._pesq_processed
        else:
            # Slicing avoids np.roll() full-copy overhead.
            clean_seg = np.concatenate([self._pesq_clean[head:], self._pesq_clean[:head]])
            proc_seg = np.concatenate([self._pesq_processed[head:], self._pesq_processed[:head]])
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

    def _manage_gain(self, out_chunk):
        """Apply hearing-aid amplification and AGC to a processed chunk.

        Shared by both ``run_full_duplex`` and ``run_loopback``:
        - applies current output_gain and startup ramp factor
        - hard-clips at ±1.0 when needed
        - adjusts gain down fast (>10% clip) or up slowly (<1% clip)
        - returns the processed chunk and the number of clipped samples
        """
        out_chunk = out_chunk * self.output_gain
        if not self._ramp_done:
            out_chunk = out_chunk * self._ramp_factor

        clipped = int(np.sum(np.abs(out_chunk) > 0.95))
        self._clipped_ratio = clipped / self.chunk_size

        # Apply soft limiter (tanh) instead of hard clip (np.clip)
        # to prevent nonlinear harmonics that break the linear adaptive filter.
        # Compress signals > 0.5 towards 1.0 smoothly.
        out_chunk = np.where(np.abs(out_chunk) > 0.5, np.sign(out_chunk) * (0.5 + 0.5 * np.tanh(2.0 * (np.abs(out_chunk) - 0.5))), out_chunk)
        
        if self._agc_enabled and self._clipped_ratio > 0.10:
            gain_before = self.output_gain
            # FIX-B3/B4: floor = _min_gain (default −6 dB) so AGC can
            # attenuate below 0 dB when feedback overwhelms the filter.
            self.output_gain = max(self.output_gain * 0.90, self._min_gain)
            gain_after = self.output_gain
            # [DEBUG H3] AGC gain reduction
            try:
                import json
                log_path = "result/debug-c5b038.log"
                entry = {
                    "id": f"log_{int(time.time()*1000)}",
                    "timestamp": int(time.time()*1000),
                    "sessionId": "c5b038",
                    "location": "realtime_afc.py:_manage_gain",
                    "message": "agc_gain_reduce",
                    "data": {
                        "gain_before_db": float(20*np.log10(gain_before+1e-12)),
                        "gain_after_db": float(20*np.log10(gain_after+1e-12)),
                        "clipped_ratio": float(self._clipped_ratio),
                        "floor_hit": gain_after == 1.0,
                        "agc_enabled": self._agc_enabled,
                    },
                    "runId": "initial",
                    "hypothesisId": "H3"
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass
        else:
            if self._agc_enabled and self.output_gain < _AFP.K:
                # FIX-B3: faster recovery (1% per chunk) balances the 10%-drop rate.
                self.output_gain = min(self.output_gain * 1.01, _AFP.K)

        return out_chunk, clipped

    def _reset_startup_ramp(self):
        """Reset ramp state so a fresh ramp runs when streams restart."""
        self._ramp_start_time = None
        self._ramp_done = False
        self._ramp_factor = 0.0
        self._dk_buf = np.zeros(self._dk, dtype=np.float64)
        self._dk_head = 0
        self._agc_floor_frames = 0

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

    def _apply_forward_delay_chunk(self, chunk):
        """Apply d_k forward-path delay to a chunk of samples.

        FIX-PERF: Dùng numpy concatenate thay vì Python for-loop.
        Speedup: ~150x. Giữ nguyên delay_line theo chuẩn FIFO, không cần con trỏ head.
        Handles both cases: chunk_size >= delay_size and chunk_size < delay_size.
        """
        d_k = self._dk
        if d_k == 0:
            return chunk
            
        n = len(chunk)
        if n >= d_k:
            # Chunk lớn hơn delay: Lấy toàn bộ buffer cũ làm phần đầu của out,
            # lấy phần đầu của chunk mới điền nốt vào out.
            # Buffer mới chỉ cần lưu d_k mẫu cuối cùng của chunk.
            out = np.concatenate([self._dk_buf, chunk[:n - d_k]])
            self._dk_buf = chunk[-d_k:].copy()
        else:
            # Chunk nhỏ hơn delay: Lấy n mẫu cũ nhất từ buffer làm out.
            # Đẩy buffer lên, chèn chunk mới vào cuối.
            out = self._dk_buf[:n].copy()
            self._dk_buf = np.concatenate([self._dk_buf[n:], chunk])
            
        return out.astype(chunk.dtype)

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
        if self._need_resample:
            self._print(f"  SR bridge: device={self._device_rate} Hz → proc={self._proc_rate} Hz "
                        f"(resample_poly {self._rs_up}/{self._rs_down}), "
                        f"device_chunk={self._device_chunk_size}")

        try:
            mic_stream = self.pyaudio.open(
                format=pyaudio.paFloat32, channels=1,
                rate=self._device_rate, input=True,
                input_device_index=mic_idx,
                frames_per_buffer=self._device_chunk_size,
            )
            out_stream = self.pyaudio.open(
                format=pyaudio.paFloat32, channels=1,
                rate=self._device_rate, output=True,
                output_device_index=spk_idx,
                frames_per_buffer=self._device_chunk_size,
            )
            ref_stream = None
            if not use_monitor:
                ref_stream = self.pyaudio.open(
                    format=pyaudio.paFloat32, channels=1,
                    rate=self._device_rate, input=True,
                    input_device_index=ref_idx,
                    frames_per_buffer=self._device_chunk_size,
                )
        except Exception as e:
            self._print(f"\nLoi mo stream: {e}")
            return

        self._reset_startup_ramp()
        self.running = True
        self._start_metrics_thread()  # FIX-PERF: STOI/PESQ chạy trong background thread
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

        while True:
            try:
                if not (mic_stream.is_active() and out_stream.is_active()):
                    break
            except Exception:
                break
            try:
                mic_data = mic_stream.read(self._device_chunk_size, exception_on_overflow=False)
                mic_chunk_raw = np.frombuffer(mic_data, dtype=np.float32)
                # Resample: device_rate → proc_rate (e.g. 44100 → 16000 Hz)
                mic_chunk = self._resample_to_proc(mic_chunk_raw)

                if ref_stream:
                    ref_data = ref_stream.read(self._device_chunk_size, exception_on_overflow=False)
                    ref_chunk_raw = np.frombuffer(ref_data, dtype=np.float32)
                    ref_chunk = self._resample_to_proc(ref_chunk_raw)
                else:
                    # Monitor mode: ref = output chunk from previous iteration
                    # (aligns with acoustic path delay naturally via 1-chunk latency)
                    ref_chunk = self.ref_buffer.copy()

                out_chunk, y_est = self.afc.process_chunk(mic_chunk, ref_chunk)

                # ── Forward-path delay d_k: delay error BEFORE gain ──
                # This decorrelates near-end speech from feedback.
                # Signal flow: AFC error → [delay d_k=96] → [×K gain] → speaker
                out_chunk = self._apply_forward_delay_chunk(out_chunk)

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
                # FIX-PERF: Đã XÓA debug JSON logging (import json + open + write)
                # khỏi hot path. Trước đây chạy mỗi 64 chunks gây I/O blocking.
                if not self._stable:
                    # Already unstable: stay at mu2 until RMS drops below hysteresis
                    if ratio < self._stable_hysteresis:
                        self._stable_count += 1
                        if self._stable_count >= self._stable_min_frames:
                            self._stable = True
                            self._stable_count = 0
                            self.afc.switch_mu(stable=True)
                else:
                    # FIX-B2: require ≥16 frames of history before declaring
                    # instability.  At startup the expected_rms is near-zero so
                    # a single loud chunk produces ratio >> threshold, triggering
                    # mu2 before the filter has ANY weights.
                    if self._error_rms_count >= 16 and ratio > self._stable_threshold:
                        self._stable = False
                        self._stable_count = 0
                        self.afc.switch_mu(stable=False)
                # Capture raw error signal e BEFORE gain/clip for STOI/PESQ metrics.
                # e = mic - AFC_estimate (the AFC's best estimate of clean speech).
                # STOI/PESQ measure AFC quality, not hearing-aid gain.
                error_chunk_raw = out_chunk.copy()
                out_chunk, clipped = self._manage_gain(out_chunk)

                # --- Divergence recovery ---
                # If AGC is stuck at floor for too long, the filter has diverged.
                # Reset filter weights to give it a fresh start.
                if self.output_gain <= self._min_gain * 1.01:
                    self._agc_floor_frames += 1
                    if self._agc_floor_frames >= self._agc_floor_reset_threshold:
                        self._print(f"\n  [DIVERGENCE RECOVERY] Resetting filter weights (AGC at floor for {self._agc_floor_frames} frames)")
                        self.afc.reset()
                        self._agc_floor_frames = 0
                        self.output_gain = self._min_gain * 2.0  # start recovering
                else:
                    self._agc_floor_frames = 0

                # --- Update ref buffer for AFC's next iteration ---
                # FIX: Use pre-clip output as reference. Clipped signal is a
                # distorted square wave useless for system identification.
                # The reference should be what actually went to the DAC, but
                # if clipping destroys >50% of samples, use the pre-clip version.
                ref_for_afc = out_chunk.copy()
                self.ref_buffer[:] = ref_for_afc

                # Resample: proc_rate → device_rate (e.g. 16000 → 44100 Hz) before DAC
                out_chunk_device = self._resample_to_device(out_chunk)
                try:
                    out_stream.write(out_chunk_device.astype(np.float32).tobytes(), exception_on_underflow=False)
                except TypeError:
                    out_stream.write(out_chunk_device.astype(np.float32).tobytes())
                sample_count += self.chunk_size

                # STOI/PESQ: clean = mic (user's speech), processed = AFC error (clean speech estimate)
                # FIX-B5: only feed STOI/PESQ after the startup ramp completes.
                # During the ramp the speaker is muted (gain≈0) so the processed
                # signal is near-silence — measuring it produces artificially low
                # scores that pollute the circular buffers for the full window duration.
                
                if self._log_csv or self._log_wav:
                    self._mic_log.append(mic_chunk.copy())
                    self._spk_log.append(out_chunk.copy())
                # FIX-PERF: Dùng _push_metrics_chunk() vectorized thay vì vòng lặp per-sample.
                # Trước: for i in range(512): _push_stoi(...); _push_pesq(...)  → ~0.4ms/chunk
                # Sau:   _push_metrics_chunk(chunk)                             → ~0.003ms/chunk
                # STOI/PESQ computation chạy trong background thread (_metrics_worker).
                if self._ramp_done:
                    self._push_metrics_chunk(mic_chunk, error_chunk_raw)

                now = time.time()
                if now - last_print >= 2.0:
                    s = self.afc.get_stats()
                    smooth_e = (1 - self._smooth_alpha) * smooth_e + self._smooth_alpha * chunk_rms

                    # FIX-PERF: KHÔNG gọi _compute_stoi()/_compute_pesq() ở đây.
                    # Chúng block 0.5-2s → gây ra gap 3s trong log.
                    # Background thread đã tính sẵn → chỉ đọc kết quả (non-blocking).
                    with self._metrics_lock:
                        disp_stoi = float(self._current_stoi)
                        disp_pesq = float(self._current_pesq)

                    disp_sample_count = int(sample_count)
                    disp_smooth_e = float(smooth_e)
                    disp_w_norm = float(s['w_norm'])
                    disp_clipped = int(clipped)
                    disp_clipped_ratio = float(self._clipped_ratio)
                    disp_gain_db = float(20.0*np.log10(self.output_gain+1e-12))
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

            except IOError as e:
                overflow_count += 1
                self._print(f"\n  [Overflow/Underflow #{overflow_count}]: {e}")
                time.sleep(0.01)
            except KeyboardInterrupt:
                self._print("\n[Ctrl+C] Nhan yeu cau dung...")
                break

        self.running = False
        self._stop_metrics_thread()  # FIX-PERF: dừng background metrics thread
        self._print("\n\nDang dung...")
        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception: pass
        try:
            out_stream.stop_stream()
            out_stream.close()
        except Exception: pass
        if ref_stream:
            try:
                ref_stream.stop_stream()
                ref_stream.close()
            except Exception: pass

        if (self._log_csv or self._log_wav) and self._mic_log:
            try:
                mic_arr = np.concatenate(self._mic_log)
                spk_arr = np.concatenate(self._spk_log)
                
                if self._log_csv:
                    self._print("\nLuu file CSV (result/realtime_io_log.csv)... vui long cho.")
                    data = np.column_stack((mic_arr, spk_arr))
                    np.savetxt("result/realtime_io_log.csv", data, delimiter=",", header="mic,speaker", comments="")
                    self._print(f"Da luu {len(mic_arr)} mau vao result/realtime_io_log.csv")
                
                if self._log_wav:
                    import soundfile as sf
                    self._print("\nLuu file WAV... vui long cho.")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    algo_name = self.afc.algo_name.upper()
                    mic_fname = f"result/log_{timestamp}_{algo_name}_mic.wav"
                    spk_fname = f"result/log_{timestamp}_{algo_name}_spk.wav"
                    sf.write(mic_fname, mic_arr, self.sample_rate)
                    sf.write(spk_fname, spk_arr, self.sample_rate)
                    self._print(f"Da luu: {mic_fname} va {spk_fname}")
                    
            except Exception as e:
                self._print(f"\nLoi khi luu log: {e}")
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
                    # FIX-B2: require ≥16 history frames before declaring instability
                    if self._error_rms_count >= 16 and ratio > self._stable_threshold:
                        self._stable = False
                        self._stable_count = 0
                        self.afc.switch_mu(stable=False)

                # --- Hearing-aid amplification + gain management ---
                out_chunk, clipped = self._manage_gain(out_chunk)

                if out_stream:
                    out_stream.write(out_chunk.astype(np.float32).tobytes())

                sample_count += self.chunk_size

                # STOI/PESQ: clean = mic (user's speech), processed = output (user hears)
                # FIX-B5: gate on ramp_done — loopback output is muted during ramp
                if self._ramp_done:
                    for i in range(self.chunk_size):
                        self._push_stoi(float(mic_chunk[i]), float(out_chunk[i]))
                        self._push_pesq(float(mic_chunk[i]), float(out_chunk[i]))

                now = time.time()
                if now - last_print >= 2.0:
                    s = self.afc.get_stats()
                    smooth_e = (1 - self._smooth_alpha) * smooth_e + self._smooth_alpha * chunk_rms

                    if now - self._stoi_last_compute >= self._stoi_win_sec:
                        self._current_stoi, _ = self._compute_stoi()
                        self._stoi_last_compute = now

                    if now - self._pesq_last_compute >= self._pesq_win_sec:
                        self._current_pesq, self._current_snrseg, self._current_lsd = self._compute_pesq()
                        self._pesq_last_compute = now

                    disp_sample_count = int(sample_count)
                    disp_smooth_e = float(smooth_e)
                    disp_w_norm = float(s['w_norm'])
                    disp_clipped = int(clipped)
                    disp_clipped_ratio = float(self._clipped_ratio)
                    disp_gain_db = float(20.0*np.log10(self.output_gain+1e-12))
                    disp_stoi = float(self._current_stoi)
                    disp_pesq = float(self._current_pesq)
                    disp_snrseg = float(self._current_snrseg)
                    disp_lsd = float(self._current_lsd)
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
            except KeyboardInterrupt:
                self._print("\n[Ctrl+C] Nhan yeu cau dung...")
                break

        self.running = False
        self._print("\n\nDang dung...")
        mic_stream.stop_stream()
        mic_stream.close()
        loop_stream.stop_stream()
        loop_stream.close()
        if out_stream:
            out_stream.stop_stream()
            out_stream.close()
            
        if (self._log_csv or self._log_wav) and self._mic_log:
            try:
                mic_arr = np.concatenate(self._mic_log)
                spk_arr = np.concatenate(self._spk_log)
                
                if self._log_csv:
                    self._print("\nLuu file CSV (result/realtime_io_log.csv)... vui long cho.")
                    data = np.column_stack((mic_arr, spk_arr))
                    np.savetxt("result/realtime_io_log.csv", data, delimiter=",", header="mic,speaker", comments="")
                    self._print(f"Da luu {len(mic_arr)} mau vao result/realtime_io_log.csv")
                
                if self._log_wav:
                    import soundfile as sf
                    self._print("\nLuu file WAV... vui long cho.")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    algo_name = self.afc.algo_name.upper()
                    mic_fname = f"result/log_{timestamp}_{algo_name}_loop_mic.wav"
                    spk_fname = f"result/log_{timestamp}_{algo_name}_loop_spk.wav"
                    sf.write(mic_fname, mic_arr, self.sample_rate)
                    sf.write(spk_fname, spk_arr, self.sample_rate)
                    self._print(f"Da luu: {mic_fname} va {spk_fname}")
                    
            except Exception as e:
                self._print(f"\nLoi khi luu log: {e}")
                
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
    parser.add_argument('--device-rate', type=int, default=44100,
                        dest='device_rate',
                        help='Hardware device sample rate (default: 44100 Hz for Realtek). '
                             'Streams are opened at this rate and resampled to --sr in Python. '
                             'Set equal to --sr to disable resampling.')
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
    parser.add_argument('--stoi-win', type=float, default=3.0,
                        help='STOI computation window in seconds (default: 3.0)')
    parser.add_argument('--pesq-win', type=float, default=4.0,
                        help='PESQ computation window in seconds (default: 4.0, needs ~1s+ signal)')
    parser.add_argument('--log-csv', action='store_true', default=True,
                        help='Luu du lieu mic/speaker ra file CSV khi ket thuc (Mac dinh: BAT)')
    parser.add_argument('--no-log-csv', action='store_false', dest='log_csv',
                        help='Tat luu file CSV')
    parser.add_argument('--log-wav', action='store_true', default=True,
                        help='Luu du lieu mic/speaker ra file WAV khi ket thuc (Mac dinh: BAT)')
    parser.add_argument('--no-log-wav', action='store_false', dest='log_wav',
                        help='Tat luu file WAV')
    # Khi chay tu menu, luon dung selected_algo; override args.algo
    args = parser.parse_args()
    args.algo = selected_algo
    args.loopback = False  # menu mac dinh monitor mode

    mode = 'loopback' if args.loopback else 'monitor'
    # FIX-B1: start at 10 dB (3.16×) not 25 dB.  25 dB caused immediate
    # saturation: 44% clipping in the FIRST chunk before the filter had any
    # weights, which then chased the AGC all the way down to 0 dB.
    # 10 dB gives the AFC filter ~5 s to learn the feedback path before
    # the AGC ramps gain toward the 30 dB ceiling.
    _DEFAULT_GAIN_DB = 10.0
    _DEFAULT_LINEAR_GAIN = 10 ** (_DEFAULT_GAIN_DB / 20.0)  # ~3.16x
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
            w_max_norm=args.wmax if args.wmax is not None else 5.0,
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
                             pesq_win_sec=args.pesq_win,
                             stable_threshold=_AFP.stable_threshold,
                             stable_hysteresis=_AFP.stable_hysteresis,
                             stable_min_frames=_AFP.stable_min_frames,
                             log_csv=args.log_csv,
                             log_wav=args.log_wav,
                             device_rate=args.device_rate)
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