import numpy as np
import threading
import time
from numba import njit
from adaptfilt.afc_params import AFCParameters as _AFP

# ------------------------------------------------------------------
# Levinson-Durbin recursion cho AR model
# ------------------------------------------------------------------
@njit(cache=True)
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
        # Vectorized inner sum: dot product of r[1:p+1] and a[1:p+1]
        num = float(np.dot(r[1:p + 1], a[1:p + 1]))
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


@njit(cache=True)
def ar_filter(sample, ar_coeffs, delay_line):
    """Lọc FIR một mẫu qua AR model: y = sum(ar_coeffs[i] * delay_line[i])."""
    delay_line[1:] = delay_line[:-1]
    delay_line[0] = sample
    if len(ar_coeffs) == 0:
        return 0.0, delay_line
    n = min(len(ar_coeffs), len(delay_line))
    # FIX-PERF: Dùng np.dot thay vì generator expression để tăng tốc O(n)
    y = np.dot(ar_coeffs[:n], delay_line[:n])
    # Clamp output to prevent overflow
    if not np.isfinite(y):
        y = 0.0
    return y, delay_line


@njit(cache=True)
def sign(x):
    """Hard signed error: sign(0) = 1 (đồng bộ MATLAB convention)."""
    return np.where(x >= 0, 1.0, -1.0)


@njit(cache=True)
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

@njit(cache=True)
def _process_chunk_numba(
    mic_chunk, ref_chunk, n, algo_id,
    gTD, TDLLs, TDLLswh, TDLMicwh, TDLLswh_d, Lswh_ap, Q_tilde_prev,
    _ref_delay_buf, ar_coeffs, ar_frame, ar_delay_mic, ar_delay_ls,
    mic_delay_buf, ls_delay_buf,
    ar_frameindex,
    M, p, mu, delta, a, lda, delta_sc, tanh_scale, w_max_norm, leaky, max_ar, 
    AR_ORDER, FRAMELENGTH, M_bs, b_hat, block_norms
):
    errors = np.zeros(n, dtype=np.float64)
    y_est = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        mic_sample = mic_chunk[i]
        ref_sample = ref_chunk[i]

        if _ref_delay_buf is not None and len(_ref_delay_buf) > 0:
            delayed_ref = _ref_delay_buf[-1]
            _ref_delay_buf[1:] = _ref_delay_buf[:-1]
            _ref_delay_buf[0] = ref_sample
            ref_sample = delayed_ref

        TDLLs[1:] = TDLLs[:-1]
        TDLLs[0] = ref_sample

        e = mic_sample - np.dot(gTD, TDLLs)
        if not np.isfinite(e):
            e = 0.0

        e_clipped = 2.0 * np.tanh(tanh_scale * e)

        mic_delay_buf[1:] = mic_delay_buf[:-1]
        mic_delay_buf[0] = mic_sample
        Micdelay = mic_delay_buf[FRAMELENGTH]

        ls_delay_buf[1:] = ls_delay_buf[:-1]
        ls_delay_buf[0] = ref_sample
        Lsdelay = ls_delay_buf[FRAMELENGTH]

        Micwh, ar_delay_mic = ar_filter(Micdelay, ar_coeffs, ar_delay_mic)
        Lswh, ar_delay_ls = ar_filter(Lsdelay, ar_coeffs, ar_delay_ls)
        if not np.isfinite(Micwh): Micwh = 0.0
        if not np.isfinite(Lswh): Lswh = 0.0

        ar_frame[1:] = ar_frame[:-1]
        ar_frame[0] = e_clipped

        if ar_frameindex == FRAMELENGTH - 1 and AR_ORDER - 1 > 0:
            R = np.zeros(AR_ORDER, dtype=np.float64)
            for j in range(AR_ORDER):
                vec_mult = np.zeros(FRAMELENGTH, dtype=np.float64)
                end_idx = min(FRAMELENGTH, FRAMELENGTH - j)
                vec_mult[:end_idx] = ar_frame[j:j + end_idx]
                R[j] = np.dot(ar_frame, vec_mult) / FRAMELENGTH

            R0 = R[0]
            if abs(R0) > 1e-12:
                R_norm = R / R0
            else:
                R_norm = R.copy()
                R_norm[0] = 1.0

            ar_new, _ = levinson_durbin(R_norm, AR_ORDER - 1, max_coeff=max_ar)
            ar_coeffs.fill(0.0)
            n_copy = min(len(ar_new), AR_ORDER)
            ar_coeffs[:n_copy] = ar_new[:n_copy]

            for j in range(len(ar_coeffs)):
                if not np.isfinite(ar_coeffs[j]):
                    ar_coeffs[j] = 0.0
                ar_coeffs[j] = max(-max_ar, min(max_ar, ar_coeffs[j]))

        ar_frameindex = (ar_frameindex + 1) % FRAMELENGTH

        TDLLswh[1:] = TDLLswh[:-1]
        TDLLswh[0] = Lswh

        ep = Micwh - np.dot(TDLLswh, gTD)
        if not np.isfinite(ep):
            ep = 0.0

        TDLMicwh[1:] = TDLMicwh[:-1]
        TDLMicwh[0] = Micwh

        TDLLswh_d[1:] = TDLLswh_d[:-1]
        TDLLswh_d[0] = Lswh

        for j in range(p):
            vec_start = j
            vec_end = j + M - 1
            if vec_end < len(TDLLswh_d):
                Lswh_ap[:, j] = TDLLswh_d[vec_start:vec_end + 1]
            else:
                avail = len(TDLLswh_d) - vec_start
                col = np.zeros(M, dtype=np.float64)
                col[:avail] = TDLLswh_d[vec_start:]
                Lswh_ap[:, j] = col

        Lswh_ap_active = np.ascontiguousarray(Lswh_ap[:, :p])
        Lswh_ap_active_T = np.ascontiguousarray(Lswh_ap_active.T)
        TDLMicwh_active = np.ascontiguousarray(TDLMicwh[:p])
        ewh_p = TDLMicwh_active - Lswh_ap_active_T @ gTD

        norm_reg = (1.0 - a) / (2.0 * M) * delta
        denom_norm = 0.0

        if algo_id == 0: # nlms
            norm_sq = np.dot(TDLLswh, TDLLswh) + delta
            u_pgs = TDLLswh * np.tanh(lda * ep)
            denom_norm = np.dot(u_pgs, u_pgs)
            update = mu * u_pgs / (norm_sq + norm_reg)

        elif algo_id == 1: # ipnlms
            gTD_abs = np.abs(gTD)
            b = (1.0 - a) / (2.0 * M) + (1.0 + a) * gTD_abs / (np.sum(gTD_abs) + delta_sc)
            u_pgs = b * TDLLswh * np.tanh(lda * ep)
            denom_norm = np.dot(u_pgs, u_pgs)
            update = mu * u_pgs / (denom_norm + norm_reg)

        elif algo_id == 2: # apa
            AtA = Lswh_ap_active_T @ Lswh_ap_active
            A_mat = AtA + delta * np.eye(p, dtype=np.float64)
            update = Lswh_ap_active @ np.linalg.solve(A_mat, np.tanh(lda * ewh_p))
            update = mu * update

        elif algo_id == 3: # ipapa
            gTD_abs = np.abs(gTD)
            b = (1.0 - a) / (2.0 * M) + (1.0 + a) * gTD_abs / (np.sum(gTD_abs) + delta_sc)
            A = Lswh_ap_active
            B_A = np.zeros_like(A)
            for j in range(p):
                B_A[:, j] = b * A[:, j]
            
            A_T = np.ascontiguousarray(A.T)
            B_A = np.ascontiguousarray(B_A)
            gram = A_T @ B_A
            gram += delta * np.eye(p, dtype=np.float64)
            update = mu * B_A @ np.linalg.solve(gram, np.tanh(lda * ewh_p))

        elif algo_id == 4: # ipapsa
            gTD_abs = np.abs(gTD)
            b = (1.0 - a) / (2.0 * M) + (1.0 + a) * gTD_abs / (np.sum(gTD_abs) + delta_sc)
            signed_err_vec = np.tanh(lda * ewh_p)
            
            g_vec = np.zeros_like(Lswh_ap_active)
            for j in range(p):
                g_vec[:, j] = b * signed_err_vec[j]
            
            g_sum = np.zeros(M, dtype=np.float64)
            for j in range(p):
                g_sum += g_vec[:, j]
                
            denom_apsp = np.dot(g_sum, g_sum) + norm_reg
            update = mu * g_sum / denom_apsp

        elif algo_id == 5: # mipapsa
            gTD_abs = np.abs(gTD)
            b = (1.0 - a) / (2.0 * M) + (1.0 + a) * gTD_abs / (np.sum(gTD_abs) + delta_sc)
            current_col = b * Lswh_ap_active[:, 0]

            Q_tilde = np.zeros((M, p), dtype=np.float64)
            Q_tilde[:, 0] = current_col
            if p >= 2:
                cols_from_prev = min(p - 1, Q_tilde_prev.shape[1])
                Q_tilde[:, 1:1 + cols_from_prev] = Q_tilde_prev[:, :cols_from_prev]

            u_tilde_pgs = Q_tilde @ np.tanh(lda * ewh_p)
            denom_norm = np.dot(u_tilde_pgs, u_tilde_pgs)
            update = mu * u_tilde_pgs / (denom_norm + norm_reg)

            if p >= 2:
                for j in range(p-1, 0, -1):
                    Q_tilde_prev[:, j] = Q_tilde_prev[:, j-1]
                Q_tilde_prev[:, 0] = current_col

        elif algo_id == 6: # bsmipapsa
            N_bs = M // M_bs
            remainder = M % M_bs
            
            for blk in range(M_bs):
                n_taps = N_bs + (1 if blk < remainder else 0)
                idx_start = blk * N_bs + min(blk, remainder)
                idx_end = idx_start + n_taps
                block_norms[blk] = np.linalg.norm(gTD[idx_start:idx_end])

            sum_block_norms = np.sum(block_norms)

            for blk in range(M_bs):
                n_taps = N_bs + (1 if blk < remainder else 0)
                idx_start = blk * N_bs + min(blk, remainder)
                idx_end = idx_start + n_taps
                b_hat_k = (1.0 - a) / (2.0 * M) + (1.0 + a) * block_norms[blk] / (2.0 * M_bs * sum_block_norms + delta_sc)
                b_hat[idx_start:idx_end] = b_hat_k

            current_col = b_hat * Lswh_ap_active[:, 0]

            Q_hat = np.zeros((M, p), dtype=np.float64)
            Q_hat[:, 0] = current_col
            if p >= 2:
                cols_from_prev = min(p - 1, Q_tilde_prev.shape[1])
                Q_hat[:, 1:1 + cols_from_prev] = Q_tilde_prev[:, :cols_from_prev]

            u_hat_pgs = Q_hat @ np.tanh(lda * ewh_p)
            denom_norm = np.dot(u_hat_pgs, u_hat_pgs)
            update = mu * u_hat_pgs / (denom_norm + norm_reg)

            if p >= 2:
                for j in range(p-1, 0, -1):
                    Q_tilde_prev[:, j] = Q_tilde_prev[:, j-1]
                Q_tilde_prev[:, 0] = current_col
        else:
            update = np.zeros(M, dtype=np.float64)

        if leaky > 0:
            gTD = (1.0 - leaky) * gTD + update
        else:
            gTD = gTD + update

        dc_mean = np.mean(gTD)
        gTD -= dc_mean

        for j in range(M):
            if not np.isfinite(gTD[j]):
                gTD[j] = 0.0

        w_norm = np.linalg.norm(gTD)
        if np.isfinite(w_norm) and w_norm > w_max_norm and w_max_norm != float('inf'):
            gTD = gTD * (w_max_norm / w_norm)

        y = np.dot(gTD, TDLLs)
        if not np.isfinite(y):
            y = 0.0

        errors[i] = e
        y_est[i] = y

    return errors, y_est, ar_frameindex

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

        # d_fb delay buffer: delays the reference signal by d_fb samples
        # before it enters TDLLs. Matches C/Simulink dg_hat_DSTATE block.
        # This decorrelates the feedback estimate from the current input.
        if d_fb > 0:
            self._ref_delay_buf = np.zeros(d_fb, dtype=np.float64)
        else:
            self._ref_delay_buf = None

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

    def process_chunk(self, mic_chunk, ref_chunk, out_dtype=np.float32):
        n = len(mic_chunk)
        algo_map = {
            'nlms': 0, 'ipnlms': 1, 'apa': 2, 'ipapa': 3,
            'ipapsa': 4, 'mipapsa': 5, 'bsmipapsa': 6
        }
        algo_id = algo_map[self.algo]
        
        if not hasattr(self, '_b_hat'):
            self._b_hat = np.zeros(self.M, dtype=np.float64)
            self._block_norms = np.zeros(self.M_bs, dtype=np.float64)

        with self._lock:
            errors, y_est, new_frameindex = _process_chunk_numba(
                mic_chunk, ref_chunk, n, algo_id,
                self.gTD, self.TDLLs, self.TDLLswh, self.TDLMicwh, self.TDLLswh_d, self.Lswh_ap, self.Q_tilde_prev,
                self._ref_delay_buf, self.ar_coeffs, self.ar_frame, self.ar_delay_mic, self.ar_delay_ls,
                self.mic_delay_buf, self.ls_delay_buf,
                self.ar_frameindex,
                self.M, self.p, self.mu, self.delta, self.a, self.lda, self.delta_sc, self.tanh_scale, self.w_max_norm, self.leaky, self.max_ar, 
                self.AR_ORDER, self.FRAMELENGTH, self.M_bs, self._b_hat, self._block_norms
            )
            self.ar_frameindex = new_frameindex

        return errors.astype(out_dtype), y_est.astype(out_dtype)
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
            if self._ref_delay_buf is not None:
                self._ref_delay_buf.fill(0.0)

    def switch_mu(self, stable):
        """Switch between nominal mu (stable) and mu2 (unstable) for HNLMS recovery.

        FIX: When stable, return to the algorithm's NOMINAL step size (mu_apa_ipapa=9e-4
        for IPAPA, not mu1=4e-6). The old code used mu1 for all algorithms when stable,
        which was ~225× too small for IPAPA and caused near-zero convergence.

        When acoustic feedback causes instability, the error magnitude spikes.
        Switching to mu2 (large step) helps the filter converge quickly to the
        new feedback path. Once stable, the nominal step ensures fine tracking.
        """
        with self._lock:
            if stable:
                # FIX: use per-algorithm nominal step (not global mu1 which is NLMS-specific)
                self.mu = _AFP.get_mu(self.algo)
            else:
                self.mu = self.mu2
            # [DEBUG H1] mu switch event
            try:
                import json, os
                log_path = "result/debug-c5b038.log"
                entry = {
                    "id": f"log_{int(time.time()*1000)}",
                    "timestamp": int(time.time()*1000),
                    "sessionId": "c5b038",
                    "location": "realtime_afc.py:switch_mu",
                    "message": "mu_switch",
                    "data": {
                        "stable": stable,
                        "mu_before": getattr(self, "_prev_mu", None),
                        "mu_after": self.mu,
                        "mu1": self.mu1,
                        "mu2": self.mu2,
                        "mu_nominal": _AFP.get_mu(self.algo),
                        "algo": self.algo,
                    },
                    "runId": "fixed_v2",
                    "hypothesisId": "H1"
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                self._prev_mu = self.mu
            except Exception:
                pass

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
