"""Tiny numpy-only module holding the pitch maths for configure_voices.py.

Why this is its own file: on Windows, multiprocessing starts every worker
process by re-importing the module the worker function lives in. If the
function lived in configure_voices.py, each worker would also import
script_agent -> faster_whisper -> ctranslate2 (several seconds, hundreds of
MB) before doing any work. Keeping the worker code here means each worker
only ever imports numpy, so the pool starts almost instantly.
"""
import numpy as np


def estimate_pitch_hz(samples: np.ndarray, sample_rate: int, fmin: int = 70, fmax: int = 300):
    """Rough fundamental-frequency estimate via autocorrelation on one
    short voiced clip - a lean, not a real pitch tracker.

    Only lags between fmin and fmax are ever used, so the autocorrelation
    is computed with an FFT (O(n log n)) rather than
    np.correlate(..., mode="full") (O(n^2)) and sliced to that window.
    """
    if len(samples) < sample_rate * 0.05:
        return None
    samples = samples.astype(np.float64)
    samples -= samples.mean()
    if np.abs(samples).max() < 1e-6:
        return None  # near-silent clip - nothing to estimate from
    min_lag, max_lag = int(sample_rate / fmax), int(sample_rate / fmin)
    if min_lag >= max_lag or max_lag >= len(samples):
        return None
    n = len(samples)
    fft_size = 1 << (2 * n - 1).bit_length()  # next pow2 >= 2n-1, avoids circular wraparound
    spectrum = np.fft.rfft(samples, n=fft_size)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[:n]
    window = corr[min_lag:max_lag]
    if window.max() <= 0:
        return None
    peak_lag = min_lag + int(np.argmax(window))
    return sample_rate / peak_lag if peak_lag > 0 else None


def clip_pitch_task(task):
    """Pool entry point. task = (speaker_name, samples, sample_rate);
    returns (speaker_name, hz_or_None). Kept as a plain top-level function
    taking/returning small picklable values so it works under Windows'
    'spawn' start method."""
    name, samples, sample_rate = task
    return name, estimate_pitch_hz(samples, sample_rate)
