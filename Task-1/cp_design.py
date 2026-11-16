"""Task 1: design of the cyclic prefix (CP) length for both scenarios.

Design rule: the CP must be at least as long as the channel memory,

    Ncp >= L - 1        (L - 1 = index of the last non-zero tap).

Then every sample inside the DFT window only sees the current OFDM symbol
(no ISI), and the linear convolution with h acts like a circular one (no ICI).

This script prints and saves the CP design of both scenarios, and plots the
ISI + ICI power that would remain if the CP were shorter than L - 1.

Run from the project root:   python3 Task-1/cp_design.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))           # for parameters.py and plot_style.py

from parameters import (CHANNEL_PROFILES, NUM_SUBCARRIERS, SAMPLE_PERIOD, SNR,
                        SUBCARRIER_SPACING)
from plot_style import BLUE, GREY, INK, ORANGE, save_figure

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"

# 5G NR normal CP for 30 kHz subcarrier spacing: 2.34 us = 36 samples at 15.36 MHz
NR_NORMAL_CP_LENGTH = 36


# ---------------------------------------------------------------------------
# Design functions
# ---------------------------------------------------------------------------
def minimum_cp_length(tap_delays):
    """Shortest CP without ISI: the delay of the last non-zero tap, L - 1."""
    return max(tap_delays)


def rms_delay_spread(tap_delays, tap_powers):
    """Power-weighted standard deviation of the tap delays (in samples)."""
    delays = np.asarray(tap_delays, dtype=float)
    weights = np.asarray(tap_powers) / np.sum(tap_powers)
    mean_delay = np.sum(weights * delays)
    return np.sqrt(np.sum(weights * (delays - mean_delay) ** 2))


def average_interference_power(tap_delays, tap_powers, cp_length, num_subcarriers):
    """Average ISI + ICI power per subcarrier when the CP is too short.

    A tap with delay l > Ncp reaches (l - Ncp) samples back into the previous
    OFDM symbol. On these samples the receiver gets the previous symbol
    instead of the cyclic copy of the current one: an error of power
    2 |h[l]|^2 per sample (two independent samples of unit power). The unitary
    DFT spreads this error energy over the N subcarriers. Averaged over the
    channel (E|h[l]|^2 = sigma_l^2):

        P_int = (2 / N) * sum_l  sigma_l^2 * max(l - Ncp, 0)
    """
    excess_delays = np.maximum(np.asarray(tap_delays) - cp_length, 0)
    return 2 / num_subcarriers * np.sum(np.asarray(tap_powers) * excess_delays)


def interference_power_per_subcarrier(taps, cp_length, num_subcarriers):
    """Exact ISI + ICI power on each subcarrier, for one channel realization.

    Valid for OFDM symbols i >= 1 (symbol 0 has nothing sent before it, so it
    sees only half of this power). Averaged over the data symbols:

        D[k] = (2 / N) * sum_{q = Ncp+1}^{L-1} |H_q[k]|^2,
        H_q[k] = sum_{l >= q} h[l] exp(-j 2 pi k l / N),

    where H_q is the response of only those taps that reach q or more samples
    back. The mean of D[k] over k is (2 / N) sum_l |h[l]|^2 max(l - Ncp, 0).
    """
    power = np.zeros(num_subcarriers)
    for first_late_tap in range(cp_length + 1, len(taps)):
        late_taps = np.zeros_like(taps)
        late_taps[first_late_tap:] = taps[first_late_tap:]
        power += np.abs(np.fft.fft(late_taps, num_subcarriers)) ** 2
    return 2 / num_subcarriers * power


def shortest_cp_below(tap_delays, tap_powers, power_limit, num_subcarriers):
    """Smallest Ncp whose average ISI + ICI power is at most power_limit."""
    cp_length = 0
    while average_interference_power(tap_delays, tap_powers, cp_length,
                                     num_subcarriers) > power_limit:
        cp_length += 1
    return cp_length


# ---------------------------------------------------------------------------
# Report table and figure
# ---------------------------------------------------------------------------
def design_summary():
    """Return the CP design of both scenarios as a list of text lines."""
    noise_power = 1 / SNR
    lines = ["Task 1: CP design (N = 512, W = 15.36 MHz, sample period 65.1 ns)", ""]

    for scenario, profile in CHANNEL_PROFILES.items():
        delays, powers = profile["tap_delays"], profile["tap_powers"]
        cp_length = minimum_cp_length(delays)
        delay_spread = rms_delay_spread(delays, powers)
        # Rule of thumb: coherence bandwidth = 1 / (5 x rms delay spread)
        coherence_bandwidth = 1 / (5 * delay_spread * SAMPLE_PERIOD)
        soft_cp = shortest_cp_below(delays, powers, noise_power / 10, NUM_SUBCARRIERS)
        nr_interference = average_interference_power(delays, powers, NR_NORMAL_CP_LENGTH,
                                                     NUM_SUBCARRIERS)
        nr_text = (f"{10 * np.log10(nr_interference):.1f} dB" if nr_interference > 0
                   else "0 (no ISI)")

        lines += [
            f"Scenario {scenario}",
            f"  sum of tap powers               : {sum(powers):.3f}",
            f"  last non-zero tap, L - 1        : {cp_length} samples "
            f"= {cp_length * SAMPLE_PERIOD * 1e6:.3f} us",
            f"  rms delay spread                : {delay_spread:.2f} samples "
            f"= {delay_spread * SAMPLE_PERIOD * 1e9:.0f} ns",
            f"  coherence bandwidth 1/(5 tau)   : {coherence_bandwidth / 1e6:.2f} MHz "
            f"= {coherence_bandwidth / SUBCARRIER_SPACING:.0f} subcarriers",
            f"  chosen CP length Ncp = L - 1    : {cp_length} samples "
            f"= {cp_length * SAMPLE_PERIOD * 1e6:.3f} us",
            f"  OFDM symbol length N + Ncp      : {NUM_SUBCARRIERS + cp_length} samples "
            f"= {(NUM_SUBCARRIERS + cp_length) * SAMPLE_PERIOD * 1e6:.2f} us",
            f"  CP overhead Ncp / (N + Ncp)     : "
            f"{100 * cp_length / (NUM_SUBCARRIERS + cp_length):.2f} %",
            f"  slot length 14 (N + Ncp)        : {14 * (NUM_SUBCARRIERS + cp_length)} samples "
            f"= {14 * (NUM_SUBCARRIERS + cp_length) * SAMPLE_PERIOD * 1e6:.1f} us",
            f"  shortest Ncp with ISI+ICI 10 dB below noise : {soft_cp} samples",
            f"  ISI+ICI with NR normal CP ({NR_NORMAL_CP_LENGTH} samples): {nr_text}",
            "",
        ]
    return lines


def plot_interference_versus_cp():
    """Average ISI + ICI power against the CP length, for both scenarios."""
    noise_power_db = 10 * np.log10(1 / SNR)
    fig, ax = plt.subplots(figsize=(7, 4))

    for scenario, color in [(1, BLUE), (2, ORANGE)]:
        profile = CHANNEL_PROFILES[scenario]
        delays, powers = profile["tap_delays"], profile["tap_powers"]
        cp_design = minimum_cp_length(delays)

        # For Ncp >= L - 1 the interference is exactly zero (minus infinity in dB)
        cp_lengths = np.arange(cp_design)
        interference = [average_interference_power(delays, powers, cp, NUM_SUBCARRIERS)
                        for cp in cp_lengths]
        ax.plot(cp_lengths, 10 * np.log10(interference), color=color, marker="o",
                markersize=3, linewidth=1.4, label=f"Scenario {scenario}")
        ax.axvline(cp_design, color=color, linestyle="--", linewidth=1.2)
        # Label to the right of the line for Scenario 1, to the left for Scenario 2
        label_x, alignment = ((cp_design + 1, "left") if scenario == 1
                              else (cp_design - 1, "right"))
        ax.text(label_x, -54.3, f"$N_{{cp}} = {cp_design}$\n(no ISI from here on)",
                color=color, fontsize=9, va="bottom", ha=alignment)

    ax.axhline(noise_power_db, color=INK, linestyle="-.", linewidth=1.1,
               label="noise power $1/\\mathrm{SNR}$ ($-30$ dB)")
    ax.axvline(NR_NORMAL_CP_LENGTH, color=GREY, linestyle=":", linewidth=1.3,
               label="5G NR normal CP (36 samples)")

    ax.set_xlim(-1, 85)
    ax.set_ylim(-55, -10)
    ax.set_xlabel("CP length $N_{cp}$ (samples)")
    ax.set_ylabel("average ISI + ICI power (dB)")
    ax.set_title("Task 1: interference left by a too short CP (average over the channel)")
    ax.grid(True)
    ax.legend(loc="upper right")
    save_figure(fig, RESULTS_FOLDER, "interference_vs_cp")


if __name__ == "__main__":
    summary = design_summary()
    print("\n".join(summary))

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    (RESULTS_FOLDER / "cp_design.txt").write_text("\n".join(summary) + "\n")
    plot_interference_versus_cp()
    print(f"Saved the summary and the figure to {RESULTS_FOLDER}")
