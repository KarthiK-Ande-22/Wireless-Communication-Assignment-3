"""Task 3: multipath channel y = h * x + w. Run this file to see both channels and check every block."""
import sys
from pathlib import Path

import numpy as np


def complex_gaussian(size, variance, rng):
    """Return independent CN(0, variance) samples; real and imaginary parts are N(0, variance / 2)."""
    return np.sqrt(variance / 2) * (rng.standard_normal(size) + 1j * rng.standard_normal(size))


def channel_taps(tap_delays, tap_powers, rng):
    """Return one tap vector h of length L = last delay + 1, with h[l] ~ CN(0, sigma_l^2) at each delay."""
    taps = np.zeros(max(tap_delays) + 1, dtype=complex)
    for delay, power in zip(tap_delays, tap_powers):
        taps[delay] = complex_gaussian(1, power, rng)[0]
    return taps


def frequency_response(taps, num_subcarriers):
    """Return H[k] = sum_l h[l] exp(-j 2 pi k l / N), which is fft(h, N)."""
    return np.fft.fft(taps, num_subcarriers)


def multipath_channel(x, taps, snr, rng):
    """Return y = h * x + w with M + L - 1 samples and noise w[m] ~ CN(0, 1/SNR)."""
    signal = np.convolve(x, taps)
    noise = complex_gaussian(len(signal), 1 / snr, rng)
    return signal + noise


# Channel realizations and check of every block

def describe_channel(scenario, tap_delays, tap_powers, seed, num_subcarriers):
    """Return the lines that show and check the channel of one scenario."""
    taps = channel_taps(tap_delays, tap_powers, np.random.default_rng(seed))
    response = frequency_response(taps, num_subcarriers)
    power_db = 10 * np.log10(np.abs(response) ** 2)

    lines = [f"Scenario {scenario}: L = {len(taps)} taps, channel seed {seed}",
             "   tap l   sigma_l^2   h[l]                 |h[l]|^2"]
    for delay, power in zip(tap_delays, tap_powers):
        h = taps[delay]
        lines.append(f"   {delay:5d}   {power:9.3f}   {h.real:+.4f} {h.imag:+.4f}j     {abs(h) ** 2:.4f}")

    # fft(h, N) must equal the sum in the definition of H[k]
    k = np.arange(num_subcarriers)[:, np.newaxis]
    l = np.arange(len(taps))[np.newaxis, :]
    response_by_sum = np.sum(taps * np.exp(-2j * np.pi * k * l / num_subcarriers), axis=1)

    # Over many draws the generator must give E|h[l]|^2 = sigma_l^2
    rng = np.random.default_rng(1)
    many_taps = np.array([channel_taps(tap_delays, tap_powers, rng) for _ in range(20000)])
    mean_power = np.mean(np.abs(many_taps[:, tap_delays]) ** 2, axis=0)

    lines += [
        f"   channel energy ||h||^2            : {np.sum(np.abs(taps) ** 2):.4f}",
        f"   mean of |H[k]|^2 (Parseval)       : {np.mean(np.abs(response) ** 2):.4f}",
        f"   max |fft(h, N) - sum formula|     : {np.max(np.abs(response - response_by_sum)):.2e}",
        f"   |H[k]|^2 min / max                : {power_db.min():.2f} dB (k = {np.argmin(power_db)})"
        f" / {power_db.max():.2f} dB (k = {np.argmax(power_db)})",
        f"   subcarriers with |H[k]|^2 < -10 dB: {np.sum(power_db < -10)}",
        "   E|h[l]|^2 over 20000 draws        : " + " ".join(f"{p:.3f}" for p in mean_power),
        "",
    ]
    return lines


def check_convolution_and_noise(taps, snr):
    """Check y against y[m] = sum_l h[l] x[m - l] and the noise variance against 1/SNR."""
    rng = np.random.default_rng(1)
    x = complex_gaussian(1000, 1.0, rng)
    y = multipath_channel(x, taps, np.inf, rng)            # 1/SNR = 0: no noise

    # The sum of the assignment, with x[m] = 0 for m < 0 and m >= M
    y_by_sum = np.zeros(len(x) + len(taps) - 1, dtype=complex)
    for delay, tap in enumerate(taps):
        y_by_sum[delay:delay + len(x)] += tap * x

    noise = multipath_channel(np.zeros(200000), np.array([1.0]), snr, rng)
    return [
        f"Convolution and noise (Scenario 2 taps, L = {len(taps)}, M = {len(x)})",
        f"   length of y                       : {len(y)} (M + L - 1 = {len(x) + len(taps) - 1})",
        f"   max |convolve(x, h) - sum formula|: {np.max(np.abs(y - y_by_sum)):.2e}",
        f"   measured noise variance x SNR     : {np.var(noise) * snr:.4f} (should be 1)",
    ]


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))   # for parameters.py
    from parameters import CHANNEL_PROFILES, CHANNEL_SEEDS, NUM_SUBCARRIERS, SNR

    # Same seeds as Task-5/main.py, so these are the channels used for the results
    lines = ["Task 3: channel realizations", ""]
    for scenario, profile in CHANNEL_PROFILES.items():
        lines += describe_channel(scenario, profile["tap_delays"], profile["tap_powers"],
                                  CHANNEL_SEEDS[scenario], NUM_SUBCARRIERS)

    profile = CHANNEL_PROFILES[2]
    taps = channel_taps(profile["tap_delays"], profile["tap_powers"],
                        np.random.default_rng(CHANNEL_SEEDS[2]))
    lines += check_convolution_and_noise(taps, SNR)

    print("\n".join(lines))
    results_folder = Path(__file__).resolve().parent / "results"
    results_folder.mkdir(exist_ok=True)
    (results_folder / "channel_check.txt").write_text("\n".join(lines) + "\n")
