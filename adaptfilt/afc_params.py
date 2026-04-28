"""
AFCParameters - All stable AFC system parameters synchronized with MATLAB/Simulink reference.

This module defines the canonical parameter values used in the MATLAB/Simulink AFC
implementation (Adpative_Filter.txt). All Python code (realtime_afc.py, afc_simulation.py,
and the adaptfilt/ library) should reference these values for consistency.
"""


class AFCParameters:
    """
    All stable AFC system parameters synchronized with MATLAB/Simulink reference.

    Parameters are organized into logical groups:
      - System:   sampling frequency, signal length, loop gain, path delays
      - Algorithm: step-sizes, regularization, proportionate-control constants
      - AR Model: pre-whitening filter dimensions and clamping limits
      - Clipping:  error-signal saturation (tanh gating)
      - Noise:     SNR/SIR, impulsive-noise statistics, probe-signal variance

    Usage
    -----
        from adaptfilt.afc_params import AFCParameters

        p = AFCParameters()
        mu = AFCParameters.get_mu('nlms')   # -> 1e-3
        mu = AFCParameters.get_mu('ipapsa')  # -> 8e-6
    """

    # ======================================================================
    # === System Parameters (Table I in MATLAB reference)
    # ======================================================================
    fs = 16000            # Hz   - Sampling frequency
    N = 960000            # samples - Total signal length (60 seconds at 16 kHz)
    Kdb = 30              # dB   - Forward-path gain (loop gain)
    K = 31.62             # -     - Linear gain = 10^(Kdb/20)
    d_k = 96              # samples - Forward-path delay
    d_fb = 1              # samples - Feedback-cancellation path delay
    Lg_hat = 64           # samples - Adaptive filter length
    La = 20               # samples - AR model filter length (pre-whitening)
    Nfreq = 512           # samples - FFT points for MIS/MSG/ASG analysis
    framelength = 160     # samples - Frame length for AR estimation (10 ms at 16 kHz)
    P = 2                 # -       - Projection order for APA/IPAPA
    M_bs = 8              # -       - Number of blocks in BSMIPAPSA
    FB_PATH_LENGTH = 100  # samples - Feedback path IR length

    # ======================================================================
    # === Step-size (mu) per algorithm group
    #   Group A: mu = 1e-3   for NLMS, IPNLMS
    #   Group B: mu = 9e-4   for APA, IPAPA
    #   Group C: mu = 8e-6   for IPAPSA, MIPAPSA, BSMIPAPSA
    #   RLS:     ffactor = 0.999 (forgetting factor)
    # ======================================================================
    mu_nlms_ipnlms = 1e-3
    mu_apa_ipapa = 9e-4
    mu_ipapsa_family = 8e-6

    # ======================================================================
    # === Algorithm Parameters (Table II in MATLAB reference)
    # ======================================================================
    # mu1: NLMS step when system is stable (sw2 mode) = mu/2
    mu1 = 4e-6
    # mu2: NLMS step when system is unstable = mu*10^4  (for HNLMS recovery)
    mu2 = 8e-2
    # delta: regularization parameter (prevents division by zero)
    delta = 1e-6
    # lda: sigmoid coefficient in signed-error term
    lda = 6
    # delta_sc: regularization for sparseness measure
    delta_sc = 1e-8
    # eps_R: epsilon in regression matrix R_mu = eps*I
    eps_R = 1e-5
    # beta: threshold factor for tanh-gated signal (threshold = beta/1000 = 0.05)
    beta = 50
    # alpha_a: proportionate algorithm control (aa=0 -> IPAPSA, aa=-1 -> APSA, aa=1 -> PAPSA)
    alpha_a = 0.5

    # ======================================================================
    # === Error Clipping / TanH Gating (Table V in MATLAB reference)
    # ======================================================================
    tanh_scale = 0.5     # e = 2*tanh(0.5*e)
    tanh_thresh = 0.15   # abs(hv - e) < 0.15 -> impulse detected -> switch to HNLMS

    # ======================================================================
    # === AR Model / Pre-whitening (Table VI in MATLAB reference)
    # ======================================================================
    # High-pass FIR filter: fir1(64, [0.025], 'high')
    #   64 taps, cutoff = 0.025 * Nyquist = ~200 Hz (removes DC / low-freq noise)
    hpf_order = 64
    hpf_cutoff = 0.025   # normalized (0-1 relative to Nyquist)
    # AR coefficient clamp limit to prevent instability
    max_ar_coeff = 6

    # ======================================================================
    # === Noise / Signal Parameters (Table III in MATLAB reference)
    # ======================================================================
    SNR = 30             # dB - Signal-to-Noise Ratio (background noise)
    SIR = 10             # dB - Signal-to-Impulsive-noise Ratio
    Var_noise = 0.001    # -     - Background white noise variance
    Var_P = 0.001        # -     - Probe signal variance
    # Impulsive noise: Bernoulli-Gaussian model
    #   n_impulsive = pB * G(0, varG)  where pB = Bernoulli probability
    Impulsive_noise_pB = 0.1    # Bernoulli probability pB = 0.1
    Impulsive_noise_varG = 1    # Gaussian variance varG = 1
    # Alpha-Stable noise parameters (commented in MATLAB, optional)
    alpha_imp = 1.8      # stability index
    gamma_imp = 1        # scale parameter

    # ======================================================================
    # === Signal selection (MATLAB: in_sig, n_sel)
    # ======================================================================
    # in_sig: input signal type
    #   0 = white noise, 1 = speech weighted noise, 2 = real speech, 3 = music
    # n_sel: noise type
    #   0 = none, 1 = WGN, 2 = babble noise (NOISEX-92), 3 = factory2, other = white (NOISEX-92)
    # prob_sig: 0 = no probe, 1 = white-noise probe
    prob_sig = 0
    in_sig = 2           # default: real speech
    n_sel = 2            # default: babble noise (NOISEX-92)

    @staticmethod
    def get_mu(algo):
        """
        Return the appropriate step-size (mu) for the given algorithm.

        Parameters
        ----------
        algo : str
            Algorithm name: 'nlms', 'ipnlms', 'apa', 'ipapa',
            'ipapsa', 'mipapsa', 'bsmipapsa'

        Returns
        -------
        float
            Step-size mu. Returns mu_ipapsa_family for unknown algorithms.
        """
        if algo in ('nlms', 'ipnlms'):
            return AFCParameters.mu_nlms_ipnlms
        elif algo in ('apa', 'ipapa'):
            return AFCParameters.mu_apa_ipapa
        else:  # ipapsa, mipapsa, bsmipapsa, or fallback
            return AFCParameters.mu_ipapsa_family
