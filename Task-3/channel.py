"""Task 3: multipath channel with additive noise.

    y[m] = sum_l h[l] x[m - l] + w[m],      w[m] ~ CN(0, 1/SNR)

The taps h[l] ~ CN(0, sigma_l^2) are drawn once and kept for the whole slot.
Running this file prints the channel realization of both scenarios and checks
every block:

    python3 Task-3/channel.py
"""
import sys
from pathlib import Path

import numpy as np


def complex_gaussian(size, variance, rng):
    """Independent CN(0, variance) samples.

    Real and imaginary parts are independent N(0, variance / 2), so that
    E|z|^2 = variance.
    """
    return np.sqrt(variance / 2) * (rng.standard_normal(size)
                                    + 1j * rng.standard_normal(size))


def channel_taps(tap_delays, tap_powers, rng):
    """One realization of the tap vector h = (h[0], ..., h[L-1]).

    Tap l = tap_delays[j] is drawn as CN(0, tap_powers[j]); all other taps are
    zero. The length is L = (largest delay) + 1.
    """
    taps = np.zeros(max(tap_delays) + 1, dtype=complex)
    for delay, power in zip(tap_delays, tap_powers):
        taps[delay] = complex_gaussian(1, power, rng)[0]
    return taps


def frequency_response(taps, num_subcarriers):
    """H[k] = sum_l h[l] exp(-j 2 pi k l / N), k = 0, ..., N - 1.

    This is the N-point DFT of the zero-padded taps, i.e. fft(h, N), without
    the factor 1/sqrt(N) (Section 4 of the assignment).
    """
    return np.fft.fft(taps, num_subcarriers)


def multipath_channel(x, taps, snr, rng):
    """Received signal y = h * x + w (linear convolution plus noise).

    y has M + L - 1 samples; the last L - 1 are the tail of the last OFDM
    symbol and are not used by the receiver.
    """
    signal = np.convolve(x, taps)
    noise = complex_gaussian(len(signal), 1 / snr, rng)
    return signal + noise


# ---------------------------------------------------------------------------
# Channel realizations and check of every block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT))       # for parameters.py
    from parameters import CHANNEL_PROFILES, CHANNEL_SEEDS, NUM_SUBCARRIERS, SNR

    lines = ["Task 3: channel realizations", ""]

    for scenario, profile in CHANNEL_PROFILES.items():
        delays, powers = profile["tap_delays"], profile["tap_powers"]

        # The same generator and seed as in Task-5/main.py, so this is
        # exactly the realization used for the results.
        seed = CHANNEL_SEEDS[scenario]
        taps = channel_taps(delays, powers, np.random.default_rng(seed))
        response = frequency_response(taps, NUM_SUBCARRIERS)

        lines.append(f"Scenario {scenario}: L = {len(taps)} taps, channel seed {seed}")
        lines.append("   tap l   sigma_l^2   h[l]                 |h[l]|^2")
        for delay, power in zip(delays, powers):
            h = taps[delay]
            lines.append(f"   {delay:5d}   {power:9.3f}   {h.real:+.4f} {h.imag:+.4f}j"
                         f"     {abs(h) ** 2:.4f}")

        # H[k] from fft must equal the sum in the definition
        k = np.arange(NUM_SUBCARRIERS)[:, np.newaxis]
        l = np.arange(len(taps))[np.newaxis, :]
        response_by_sum = np.sum(taps * np.exp(-2j * np.pi * k * l / NUM_SUBCARRIERS), axis=1)
        power_db = 10 * np.log10(np.abs(response) ** 2)

        lines += [
            f"   channel energy ||h||^2           : {np.sum(np.abs(taps) ** 2):.4f}",
            f"   mean of |H[k]|^2 over k          : {np.mean(np.abs(response) ** 2):.4f}"
            "  (equal to ||h||^2, Parseval)",
            f"   max |fft(h, N) - sum formula|    : "
            f"{np.max(np.abs(response - response_by_sum)):.2e}",
            f"   |H[k]|^2 min / max               : {power_db.min():.2f} dB"
            f" (k = {np.argmin(power_db)}) / {power_db.max():.2f} dB (k = {np.argmax(power_db)})",
            f"   subcarriers with |H[k]|^2 < -10 dB: {np.sum(power_db < -10)}",
        ]

        # The tap generator must give E|h[l]|^2 = sigma_l^2 on average
        rng = np.random.default_rng(1)
        many_taps = np.array([channel_taps(delays, powers, rng) for _ in range(20000)])
        mean_power = np.mean(np.abs(many_taps[:, delays]) ** 2, axis=0)
        lines.append("   E|h[l]|^2 over 20000 draws      : "
                     + " ".join(f"{p:.3f}" for p in mean_power))
        lines.append("")

    # Convolution length and noise variance
    rng = np.random.default_rng(1)
    x = complex_gaussian(1000, 1.0, rng)
    taps = np.array([1.0, 0.0, 0.5j])
    y = multipath_channel(x, taps, SNR, rng)
    noise_only = multipath_channel(np.zeros(200000), np.array([1.0]), SNR, rng)
    lines += [
        "Convolution and noise",
        f"   length of y for M = 1000, L = 3  : {len(y)}  (M + L - 1 = 1002)",
        f"   measured noise variance x SNR    : {np.var(noise_only) * SNR:.4f}  (should be 1)",
    ]

    print("\n".join(lines))
    results_folder = Path(__file__).resolve().parent / "results"
    results_folder.mkdir(exist_ok=True)
    (results_folder / "channel_check.txt").write_text("\n".join(lines) + "\n")
