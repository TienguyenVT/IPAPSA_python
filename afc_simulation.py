"""
AFC Offline Simulation — benchmark against C/Simulink reference
================================================================

Reads Speech_data.csv (960,000 samples at 16 kHz = 60 s), simulates the
feedback loop with the known feedback path from the C reference system,
runs the selected adaptive algorithm, and evaluates STOI / PESQ vs the
clean speech reference.

Usage:
    python afc_simulation.py --algo ipapa
    python afc_simulation.py --algo nlms --gain 20
    python afc_simulation.py --all            # run all 7 algorithms
"""

import numpy as np
import os
import sys
import time
import argparse

# Allow running directly
_sys_path_dir = os.path.dirname(os.path.abspath(__file__))
if _sys_path_dir not in sys.path:
    sys.path.insert(0, _sys_path_dir)

from adaptfilt.afc_params import AFCParameters as _AFP
from adaptive_algorithms import AdaptiveFilter

# Optional quality metrics
try:
    import pystoi
    _HAS_PYSTOI = True
except ImportError:
    _HAS_PYSTOI = False

try:
    from pesq import pesq as _pesq_official
    _HAS_PESQ = True
except ImportError:
    _HAS_PESQ = False

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

# =====================================================================
# Feedback path impulse responses from PEM_AFC_2019a_data.c
# pooled2 = g   (initial feedback path, 100 taps)
# pooled3 = gc  (changed feedback path, switched at t/2)
# =====================================================================
FEEDBACK_PATH_G = np.array([
    8.458E-6, -3.9859E-5, 7.7148E-5, -0.00013507, 0.00035625, -0.00042979,
    0.00017079, 0.00014615, 0.0074257, -0.0075748, -0.0085036, 0.005697,
    -0.0025054, 0.013486, 0.0047647, -0.023113, -0.010144, 0.026643, 0.013986,
    -0.024217, -0.0093478, 0.014536, 0.0081798, -0.007059, -0.0088653, 0.0032595,
    0.0080405, -0.00021787, -0.0079156, -0.0018351, 0.0051693, 0.0023075,
    -0.0010989, -0.0028856, -0.0013722, 0.0026684, 0.0024244, -0.0017269,
    -0.002177, 0.0015355, 0.0016394, -0.00080943, -0.0012856, -8.3865E-6,
    0.0012062, 0.00044427, -0.0010591, -0.00085134, 0.00053259, 0.00063423,
    -0.00022863, -0.00032311, -5.3609E-5, 0.00015773, 0.00023724, -9.5017E-6,
    -0.00014603, 0.00012564, 0.00029421, -2.2141E-5, -0.00021388, -9.6318E-6,
    0.000136, 8.3777E-5, -7.4857E-5, -0.00019377, -8.1688E-5, 0.00013548,
    7.9803E-5, -0.00013696, -8.5435E-5, 9.949E-5, 8.5649E-5, -4.3674E-5,
    -3.6028E-5, 5.7897E-5, 9.1087E-5, 3.3219E-5, -4.1812E-5, -3.6778E-5,
    1.7172E-5, 3.6447E-5, -1.1983E-5, -4.525E-5, -2.8653E-5, -4.7986E-6,
    -7.7795E-6, -2.6756E-5, -1.6189E-5, -6.6439E-7, 2.0187E-5, 9.2392E-6,
    -1.726E-5, 7.1136E-6, 3.7816E-5, 5.9541E-6, -2.3292E-5, 2.7236E-5, 2.5848E-5,
    -4.4717E-5
], dtype=np.float64)

FEEDBACK_PATH_GC = np.array([
    1.485E-5, -2.8707E-5, 4.4561E-5, -0.00010223, 0.00032592, -0.00031701,
    -0.00011651, 0.00083057, 0.0089604, -0.0075342, -0.015697, 0.0077682,
    0.001448, 0.01306, 0.012987, -0.033363, -0.019824, 0.037109, 0.02535,
    -0.032846, -0.019682, 0.023896, 0.013636, -0.014405, -0.014514, 0.010775,
    0.014708, -0.0075072, -0.014597, 0.0022154, 0.012219, -9.1474E-5, -0.0064207,
    -0.00082208, 0.00085318, 0.0013965, 0.0016699, -0.00040258, -0.0018229,
    0.00057796, 0.0014366, -0.00069711, -0.00068471, 0.00013155, 0.00061773,
    0.00020191, -0.00081295, -0.00061432, 0.00033541, 0.00044308, -0.00030051,
    4.0321E-5, 0.00015653, -0.00041205, 4.0685E-5, 0.00055365, 0.00014273,
    -0.00033053, 2.6829E-6, 0.000314, 4.0536E-5, -0.00022689, -4.4977E-5,
    0.00015014, -5.9E-5, -0.00020473, -0.00013106, 2.6647E-5, 0.0001627,
    -1.2268E-5, -9.926E-5, 5.4174E-6, 0.00011784, -0.00011385, -0.00014156,
    0.00027575, 0.00023404, -0.00014693, -0.00019171, 0.00015063, 2.4505E-5,
    -0.00026046, 7.1684E-5, 0.00032742, 5.2541E-5, -0.00047971, -0.00027189,
    0.00033133, 0.00023919, -8.9467E-5, -9.3081E-5, 1.3084E-5, -0.00014543,
    2.8379E-6, 0.00028511, 5.4471E-6, -0.00026908, 8.8607E-5, 0.00038462,
    -0.00032589
], dtype=np.float64)


def load_speech_data(csv_path):
    """Load speech signal from CSV (one sample per line)."""
    print(f"  Loading speech data: {csv_path}")
    data = np.loadtxt(csv_path, dtype=np.float64)
    print(f"  Loaded {len(data)} samples ({len(data)/_AFP.fs:.1f}s at {_AFP.fs} Hz)")
    return data


def run_simulation(speech, algo='ipapa', gain_db=None, switch_path=True,
                   snr_db=None, verbose=True, filter_len=None,
                   clip_speaker=False, step_override=None):
    """
    Run AFC simulation with known feedback path.

    Parameters
    ----------
    speech : ndarray
        Clean speech signal (960,000 samples at 16 kHz).
    algo : str
        Algorithm name ('nlms', 'ipnlms', 'apa', 'ipapa', etc.)
    gain_db : float or None
        Forward-path gain in dB (default: AFCParameters.Kdb = 30 dB).
    switch_path : bool
        If True, switch feedback path from g to gc at t/2 (like Simulink).
    snr_db : float or None
        If given, add white noise at this SNR.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys: 'output', 'error', 'speaker_out', 'mic_signal',
                    'clean_delayed', 'w_norm_history', 'algo', 'stats'
    """
    N = len(speech)
    fs = _AFP.fs
    d_k = _AFP.d_k        # = 96 samples forward-path delay
    d_fb = _AFP.d_fb       # = 1 sample feedback-path delay
    M = _AFP.Lg_hat        # = 100 taps (updated)
    K_db = gain_db if gain_db is not None else _AFP.Kdb
    K = 10 ** (K_db / 20.0)
    fb_path_len = len(FEEDBACK_PATH_G)  # = 100
    M = filter_len if filter_len else _AFP.Lg_hat

    if verbose:
        print(f"\n{'='*60}")
        print(f"  AFC Simulation: {algo.upper()}")
        print(f"  Samples: {N} ({N/fs:.1f}s)")
        print(f"  Gain K: {K_db:.1f} dB ({K:.2f}x)")
        print(f"  d_k: {d_k}, d_fb: {d_fb}, M: {M}")
        print(f"  Path switch at t/2: {switch_path}")
        print(f"{'='*60}")

    # Create adaptive filter
    step_arg = step_override if step_override else None
    afc = AdaptiveFilter(
        M=M, algo=algo, step=step_arg, delta=_AFP.delta,
        a=_AFP.alpha_a, p=_AFP.P,
        mu1=_AFP.mu1, mu2=_AFP.mu2, eps=_AFP.eps,
        lda=_AFP.lda, delta_sc=_AFP.delta_sc,
        beta=_AFP.beta, M_bs=_AFP.M_bs,
        tanh_scale=_AFP.tanh_scale, tanh_thresh=_AFP.tanh_thresh,
        max_ar=_AFP.max_ar_coeff, fs=fs, d_k=d_k, d_fb=d_fb,
    )
    if verbose:
        print(f"  mu (step): {afc.mu:.6f}")

    # Noise
    if snr_db is not None:
        sig_power = np.mean(speech ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power) * np.random.randn(N)
    else:
        noise = np.zeros(N, dtype=np.float64)

    # Buffers
    speaker_delay_buf = np.zeros(d_k, dtype=np.float64)  # forward-path delay
    dk_head = 0
    fb_delay_line = np.zeros(fb_path_len, dtype=np.float64)  # acoustic feedback path delay line

    # Current feedback path
    g_current = FEEDBACK_PATH_G.copy()
    switch_sample = N // 2 if switch_path else N + 1

    # Output arrays
    error_out = np.zeros(N, dtype=np.float64)
    speaker_out = np.zeros(N, dtype=np.float64)
    mic_signal = np.zeros(N, dtype=np.float64)
    clean_delayed = np.zeros(N, dtype=np.float64)
    n_hist = N // 1000
    w_norm_hist = np.zeros(n_hist, dtype=np.float64)

    t_start = time.time()
    last_print = t_start

    for n in range(N):
        # Switch feedback path at t/2
        if switch_path and n == switch_sample:
            g_current = FEEDBACK_PATH_GC.copy()
            if verbose:
                print(f"\n  [t/2] Feedback path switched at sample {n}")

        # 1. Compute acoustic feedback: fb = g * speaker_delay_line
        fb = np.dot(g_current, fb_delay_line)
        if not np.isfinite(fb):
            fb = 0.0

        # 2. Mic signal = clean speech + feedback + noise
        x = speech[n] + fb + noise[n]
        if not np.isfinite(x):
            x = 0.0
        mic_signal[n] = x

        # 3. AFC: error = mic - estimated_feedback
        # ref = previous speaker output (from fb_delay_line[0])
        ref = fb_delay_line[0] if n > 0 else 0.0
        e, y_hat = afc.process_sample(x, ref)
        if not np.isfinite(e):
            e = 0.0
        error_out[n] = e

        # 4. Forward-path delay d_k
        delayed_e = speaker_delay_buf[dk_head]
        speaker_delay_buf[dk_head] = e
        dk_head = (dk_head + 1) % d_k

        # Store the clean speech at the same delay for comparison
        if n >= d_k:
            clean_delayed[n] = speech[n - d_k]

        # 5. Gain: speaker output = K * delayed_error
        # Soft limiter by default: tanh prevents overflow while preserving
        # smooth signal structure. Hard clip creates harmonics that break
        # the linear adaptive filter's convergence.
        y = K * delayed_e
        if clip_speaker:
            y = max(-1.0, min(1.0, y))  # Hard clip (simulate real DAC)
        else:
            # Soft limiter: smooth tanh compression
            # Preserves small-signal linearity, compresses large signals
            if abs(y) > 0.5:
                y = np.tanh(y)
        speaker_out[n] = y

        # 6. Update acoustic feedback delay line
        fb_delay_line[1:] = fb_delay_line[:-1]
        fb_delay_line[0] = y

        # Progress
        if n % 1000 == 0 and n // 1000 < n_hist:
            w_norm_hist[n // 1000] = np.linalg.norm(afc.gTD)

        if verbose and time.time() - last_print >= 2.0:
            pct = 100.0 * n / N
            w_norm = np.linalg.norm(afc.gTD)
            elapsed = time.time() - t_start
            eta = elapsed / max(n, 1) * (N - n)
            print(f"  [{pct:5.1f}%] sample {n:>9,}/{N:,} | "
                  f"W_norm: {w_norm:.6f} | "
                  f"e_rms: {np.sqrt(np.mean(error_out[max(0,n-1000):n+1]**2)):.6f} | "
                  f"ETA: {eta:.0f}s", end='\r')
            last_print = time.time()

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  Done in {elapsed:.1f}s ({N/elapsed:.0f} samples/s)")

    return {
        'output': error_out,
        'speaker_out': speaker_out,
        'mic_signal': mic_signal,
        'clean_delayed': clean_delayed,
        'clean_original': speech,
        'w_norm_history': w_norm_hist,
        'final_gTD': afc.gTD.copy(),
        'algo': algo,
        'elapsed': elapsed,
        'K_db': K_db,
    }


def evaluate_quality(result, verbose=True):
    """Compute STOI, PESQ, SNRseg from simulation result."""
    fs = _AFP.fs
    d_k = _AFP.d_k

    # Align signals: clean_delayed vs error_out (both delayed by d_k)
    # Skip the first 2*d_k samples (transient) and last d_k samples
    start = 2 * d_k
    end = len(result['output']) - d_k
    clean = result['clean_delayed'][start:end]
    processed = result['output'][start:end]

    # Normalize
    clean_max = np.max(np.abs(clean)) + 1e-12
    clean_norm = clean / clean_max
    proc_norm = processed / clean_max

    metrics = {'algo': result['algo'], 'K_db': result['K_db']}

    # STOI
    if _HAS_PYSTOI:
        try:
            stoi_val = pystoi.stoi(clean_norm, proc_norm, fs, extended=False)
            metrics['stoi'] = float(stoi_val)
        except Exception as ex:
            metrics['stoi'] = f"error: {ex}"
    else:
        metrics['stoi'] = 'pystoi not installed'

    # PESQ
    if _HAS_PESQ:
        try:
            mode = 'wb' if fs == 16000 else 'nb'
            pesq_val = _pesq_official(fs, clean_norm.astype(np.float32),
                                       proc_norm.astype(np.float32), mode)
            metrics['pesq'] = float(pesq_val)
        except Exception as ex:
            metrics['pesq'] = f"error: {ex}"
    else:
        metrics['pesq'] = 'pesq not installed'

    # Segmental SNR
    frame_len = int(0.03 * fs)
    hop = frame_len // 2
    snr_vals = []
    for i in range(0, len(clean) - frame_len, hop):
        r = clean[i:i+frame_len]
        d = processed[i:i+frame_len]
        P_sig = np.sum(r**2) + 1e-12
        P_noise = np.sum((d - r)**2) + 1e-12
        snr = 10 * np.log10(P_sig / P_noise)
        if -10 < snr < 35:
            snr_vals.append(snr)
    metrics['snrseg'] = float(np.mean(snr_vals)) if snr_vals else 0.0

    # ERLE: Echo Return Loss Enhancement
    # ERLE = 10*log10(E[mic^2] / E[error^2])
    mic_power = np.mean(result['mic_signal'][start:end]**2) + 1e-12
    err_power = np.mean(processed**2) + 1e-12
    metrics['erle'] = float(10 * np.log10(mic_power / err_power))

    # Misadjustment: compare estimated filter vs true feedback path
    final_gTD = result['final_gTD']
    metrics['final_w_norm'] = float(np.linalg.norm(final_gTD))

    # Misadjustment = ||g_hat - g_true||^2 / ||g_true||^2
    g_true_norm = np.linalg.norm(FEEDBACK_PATH_G)
    if g_true_norm > 1e-12:
        # Pad or truncate to match lengths
        g_len = min(len(final_gTD), len(FEEDBACK_PATH_G))
        mis = np.linalg.norm(final_gTD[:g_len] - FEEDBACK_PATH_G[:g_len]) / g_true_norm
        metrics['misadjustment'] = float(mis)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Quality Metrics: {result['algo'].upper()}")
        print(f"{'='*60}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:>15s}: {v:.4f}")
            else:
                print(f"  {k:>15s}: {v}")
        print(f"{'='*60}")

    return metrics


def save_wav(filename, signal, fs=16000):
    """Save signal as WAV file."""
    if _HAS_SF:
        # Normalize to [-1, 1]
        peak = np.max(np.abs(signal)) + 1e-12
        sf.write(filename, (signal / peak).astype(np.float32), fs)
        print(f"  Saved: {filename}")
    else:
        # Fallback: raw binary
        raw_path = filename.replace('.wav', '.raw')
        signal.astype(np.float32).tofile(raw_path)
        print(f"  Saved (raw float32): {raw_path}")
        print(f"  (install soundfile for WAV: pip install soundfile)")


def main():
    parser = argparse.ArgumentParser(description='AFC Offline Simulation')
    parser.add_argument('--algo', type=str, default='ipapa',
                        choices=list(AdaptiveFilter.ALGORITHMS.keys()),
                        help='Algorithm (default: ipapa)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 7 algorithms')
    parser.add_argument('--gain', type=float, default=None,
                        help=f'Gain in dB (default: {_AFP.Kdb})')
    parser.add_argument('--no-switch', action='store_true',
                        help='Do NOT switch feedback path at t/2')
    parser.add_argument('--snr', type=float, default=None,
                        help='Add white noise at this SNR (dB)')
    parser.add_argument('--speech', type=str, default='Speech_data.csv',
                        help='Input speech CSV file')
    parser.add_argument('--out_dir', type=str, default='result/sim_output',
                        help='Output directory for metrics and audio')
    parser.add_argument('--save-wav', action='store_true',
                        help='Save output WAV files')
    parser.add_argument('--clip', action='store_true',
                        help='Clip speaker output to ±1 (simulate real DAC)')
    parser.add_argument('--filter-len', type=int, default=None,
                        help='Adaptive filter length (default: AFCParameters.Lg_hat)')
    parser.add_argument('--step', type=float, default=None,
                        help='Override step size (mu) for all algorithms')
    args = parser.parse_args()

    # Load speech
    speech_path = os.path.join(_sys_path_dir, args.speech)
    if not os.path.exists(speech_path):
        print(f"ERROR: Speech file not found: {speech_path}")
        sys.exit(1)
    speech = load_speech_data(speech_path)

    # Algorithms to run
    if args.all:
        algo_list = list(AdaptiveFilter.ALGORITHMS.keys())
    else:
        algo_list = [args.algo]

    all_metrics = []

    for algo in algo_list:
        result = run_simulation(
            speech, algo=algo,
            gain_db=args.gain,
            switch_path=not args.no_switch,
            snr_db=args.snr,
            filter_len=args.filter_len,
            clip_speaker=args.clip,
            step_override=args.step,
        )
        metrics = evaluate_quality(result)
        all_metrics.append(metrics)

        if args.save_wav:
            out_dir = os.path.join(_sys_path_dir, args.out_dir)
            os.makedirs(out_dir, exist_ok=True)
            save_wav(os.path.join(out_dir, f'{algo}_output.wav'),
                     result['output'], _AFP.fs)
            save_wav(os.path.join(out_dir, f'{algo}_speaker.wav'),
                     result['speaker_out'], _AFP.fs)
            save_wav(os.path.join(out_dir, f'{algo}_mic.wav'),
                     result['mic_signal'], _AFP.fs)

    # Summary table
    if len(all_metrics) > 1:
        print(f"\n{'='*80}")
        print(f"  SUMMARY — All Algorithms")
        print(f"{'='*80}")
        print(f"  {'Algorithm':<12s} | {'STOI':>8s} | {'PESQ':>8s} | {'SNRseg':>8s} | {'ERLE':>8s} | {'W_norm':>8s}")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for m in all_metrics:
            stoi_s = f"{m['stoi']:.4f}" if isinstance(m['stoi'], float) else 'N/A'
            pesq_s = f"{m['pesq']:.4f}" if isinstance(m['pesq'], float) else 'N/A'
            print(f"  {m['algo']:<12s} | {stoi_s:>8s} | {pesq_s:>8s} | "
                  f"{m['snrseg']:8.2f} | {m['erle']:8.2f} | {m['final_w_norm']:8.4f}")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
