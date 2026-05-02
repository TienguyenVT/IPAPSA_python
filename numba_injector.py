import os

with open('adaptive_algorithms.py', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

# 1. Add numba import
new_lines = []
for l in lines:
    new_lines.append(l)
    if l.startswith('import time'):
        new_lines.append('from numba import njit')

# 2. Add @njit to utils
def add_njit(func_name):
    for i, l in enumerate(new_lines):
        if l.startswith(f'def {func_name}('):
            new_lines.insert(i, '@njit(cache=True)')
            break

add_njit('levinson_durbin')
add_njit('ar_filter')
add_njit('sign')
add_njit('soft_sign')

# 3. Create the massive Numba function
numba_func = """
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
    
    # Pre-allocate for bsmipapsa
    b = np.zeros(M, dtype=np.float64)
    
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

        Lswh_ap_active = Lswh_ap[:, :p]
        TDLMicwh_active = TDLMicwh[:p]
        ewh_p = TDLMicwh_active - Lswh_ap_active.T @ gTD

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
            AtA = Lswh_ap_active.T @ Lswh_ap_active
            A_mat = AtA + delta * np.eye(p, dtype=np.float64)
            update = Lswh_ap_active @ np.linalg.solve(A_mat, np.tanh(lda * ewh_p))
            update = mu * update

        elif algo_id == 3: # ipapa
            gTD_abs = np.abs(gTD)
            b = (1.0 - a) / (2.0 * M) + (1.0 + a) * gTD_abs / (np.sum(gTD_abs) + delta_sc)
            A = Lswh_ap_active
            # np.newaxis and broadcasting in numba
            B_A = np.zeros_like(A)
            for j in range(p):
                B_A[:, j] = b * A[:, j]
            
            gram = A.T @ B_A
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

            if p >= 2:
                Q_tilde = np.zeros((M, p), dtype=np.float64)
                Q_tilde[:, 0] = current_col
                cols_from_prev = min(p - 1, Q_tilde_prev.shape[1])
                Q_tilde[:, 1:1 + cols_from_prev] = Q_tilde_prev[:, :cols_from_prev]
            else:
                Q_tilde = current_col.copy()

            if p >= 2:
                u_tilde_pgs = Q_tilde @ np.tanh(lda * ewh_p)
            else:
                u_tilde_pgs = Q_tilde * np.tanh(lda * ewh_p)[0]
                
            denom_norm = np.dot(u_tilde_pgs, u_tilde_pgs)
            update = mu * u_tilde_pgs / (denom_norm + norm_reg)

            if p >= 2:
                # Numba compatible shift
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

            if p >= 2:
                Q_hat = np.zeros((M, p), dtype=np.float64)
                Q_hat[:, 0] = current_col
                cols_from_prev = min(p - 1, Q_tilde_prev.shape[1])
                Q_hat[:, 1:1 + cols_from_prev] = Q_tilde_prev[:, :cols_from_prev]
            else:
                Q_hat = current_col.copy()

            if p >= 2:
                u_hat_pgs = Q_hat @ np.tanh(lda * ewh_p)
            else:
                u_hat_pgs = Q_hat * np.tanh(lda * ewh_p)[0]

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
"""

new_lines.insert(-1, numba_func)

with open('adaptive_algorithms_numba_temp.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))
