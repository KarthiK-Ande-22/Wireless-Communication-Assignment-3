"""Task 1: design of the cyclic prefix (CP) length for both scenarios.

Design rule: the CP must be at least as long as the channel memory,

    Ncp >= L - 1        (L - 1 = index of the last non-zero tap).

Then every sample inside the DFT window only sees the current OFDM symbol
(no ISI), and the linear convolution with h acts like a circular one (no ICI).
The same Ncp is used for all 14 OFDM symbols of a slot.

This script
  1. prints and saves the CP design of both scenarios,
  2. compares candidate CP lengths: overhead and energy cost against the
     ISI + ICI power and the SINR they leave,
  3. checks the ISI + ICI formula by Monte Carlo simulation of the whole link
     (Tasks 2-4) over many random channel realizations,
  4. plots the ISI + ICI power and the SINR against the CP length.

Run from the project root:   python3 Task-1/cp_design.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# parameters.py and plot_style.py are in the project root; the blocks of the
# link (used only for the Monte Carlo check) are in the folders of Tasks 2-4
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for folder in ["", "Task-2", "Task-3", "Task-4"]:
    sys.path.append(str(PROJECT_ROOT / folder))

from parameters import (CHANNEL_PROFILES, CP_CHECK_SEED, NUM_DATA_SYMBOLS, NUM_OFDM_SYMBOLS,
                        NUM_SUBCARRIERS, SAMPLE_PERIOD, SNR, SUBCARRIER_SPACING)
from plot_style import BLUE, GREY, INK, ORANGE, save_figure
from transmitter import generate_data, ofdm_transmitter
from channel import channel_taps, frequency_response, multipath_channel
from receiver import ofdm_receiver

RESULTS_FOLDER = Path(__file__).resolve().parents[1] / "results" / "Task-1"

# 5G NR normal CP for 30 kHz subcarrier spacing: 2.34 us = 36 samples at 15.36 MHz
NR_NORMAL_CP_LENGTH = 36

# CP lengths compared in the design table
CANDIDATE_CP_LENGTHS = {1: [2, 3, 4], 2: [36, 55, 69, 75]}

# Monte Carlo check: number of random channel realizations and the CP lengths simulated
NUM_CHANNEL_REALIZATIONS = 1000
SIMULATED_CP_LENGTHS = {
    1: [0, 1, 2, 3, 4, 8],
    2: [0, 4, 10, 20, 30, 36, 45, 55, 60, 65, 69, 72, 74, 75, 80],
}


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


def average_sinr(tap_delays, tap_powers, cp_length, num_subcarriers):
    """Average SINR per subcarrier: E|H[k]|^2 / (P_int + 1/SNR), with E|H[k]|^2 = 1."""
    signal_power = np.sum(tap_powers)
    interference = average_interference_power(tap_delays, tap_powers, cp_length,
                                              num_subcarriers)
    return signal_power / (interference + 1 / SNR)


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


def db(value):
    """10 log10(value)."""
    return 10 * np.log10(value)


# ---------------------------------------------------------------------------
# Monte Carlo check of P_int with the link of Tasks 2-4
# ---------------------------------------------------------------------------
def simulate_average_interference(scenario, cp_lengths, num_realizations, rng):
    """Measured ISI + ICI power and signal power, averaged over many channels.

    For every realization, new data and new channel taps are drawn, and the
    slot is sent through the transmitter, channel and receiver without noise
    for every CP length (the same data and channel for all CP lengths). The
    disturbance y~_i[k] - H[k] d~_i[k] is then only ISI + ICI. Averages use
    the OFDM symbols 1-13 (nothing is sent before symbol 0).
    """
    profile = CHANNEL_PROFILES[scenario]
    no_noise_snr = np.inf
    interference = np.zeros(len(cp_lengths))
    signal = np.zeros(len(cp_lengths))

    for _ in range(num_realizations):
        data = generate_data(NUM_DATA_SYMBOLS, rng)
        taps = channel_taps(profile["tap_delays"], profile["tap_powers"], rng)
        response = frequency_response(taps, NUM_SUBCARRIERS)

        for index, cp_length in enumerate(cp_lengths):
            x, grid = ofdm_transmitter(data, NUM_SUBCARRIERS, cp_length)
            y = multipath_channel(x, taps, no_noise_snr, rng)
            _, received_grid, _ = ofdm_receiver(y, response, NUM_SUBCARRIERS, cp_length,
                                                NUM_OFDM_SYMBOLS)
            wanted = response[:, np.newaxis] * grid
            interference[index] += np.mean(np.abs(received_grid - wanted)[:, 1:] ** 2)
            signal[index] += np.mean(np.abs(wanted[:, 1:]) ** 2)

    return interference / num_realizations, signal / num_realizations


# ---------------------------------------------------------------------------
# Report tables and figure
# ---------------------------------------------------------------------------
def design_summary():
    """The CP design of both scenarios, as a list of text lines."""
    noise_power = 1 / SNR
    lines = ["Task 1: CP design (N = 512, W = 15.36 MHz, sample period 65.1 ns)",
             "The same Ncp is used for all 14 OFDM symbols of a slot.", ""]

    for scenario, profile in CHANNEL_PROFILES.items():
        delays, powers = profile["tap_delays"], profile["tap_powers"]
        cp_length = minimum_cp_length(delays)
        delay_spread = rms_delay_spread(delays, powers)
        # Rule of thumb: coherence bandwidth = 1 / (5 x rms delay spread)
        coherence_bandwidth = 1 / (5 * delay_spread * SAMPLE_PERIOD)
        soft_cp = shortest_cp_below(delays, powers, noise_power / 10, NUM_SUBCARRIERS)
        nr_interference = average_interference_power(delays, powers, NR_NORMAL_CP_LENGTH,
                                                     NUM_SUBCARRIERS)
        nr_text = f"{db(nr_interference):.1f} dB" if nr_interference > 0 else "0 (no ISI)"
        block_length = NUM_SUBCARRIERS + cp_length

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
            f"  OFDM symbol length N + Ncp      : {block_length} samples "
            f"= {block_length * SAMPLE_PERIOD * 1e6:.2f} us",
            f"  CP overhead Ncp / (N + Ncp)     : {100 * cp_length / block_length:.2f} %",
            f"  CP energy loss (N + Ncp) / N    : {db(block_length / NUM_SUBCARRIERS):.2f} dB",
            f"  slot length 14 (N + Ncp)        : {14 * block_length} samples "
            f"= {14 * block_length * SAMPLE_PERIOD * 1e6:.1f} us",
            f"  shortest Ncp with ISI+ICI 10 dB below noise : {soft_cp} samples",
            f"  ISI+ICI with NR normal CP ({NR_NORMAL_CP_LENGTH} samples): {nr_text}",
            "",
        ]
    return lines


def candidate_summary():
    """Cost and benefit of candidate CP lengths (average over the channel)."""
    lines = ["Candidate CP lengths (average over the channel, SNR = 30 dB)",
             "   Ncp   us      overhead   energy loss   ISI+ICI     SINR      SINR loss"]
    for scenario, profile in CHANNEL_PROFILES.items():
        delays, powers = profile["tap_delays"], profile["tap_powers"]
        lines.append(f"Scenario {scenario}")
        for cp_length in CANDIDATE_CP_LENGTHS[scenario]:
            block_length = NUM_SUBCARRIERS + cp_length
            interference = average_interference_power(delays, powers, cp_length,
                                                      NUM_SUBCARRIERS)
            sinr_db = db(average_sinr(delays, powers, cp_length, NUM_SUBCARRIERS))
            interference_text = f"{db(interference):6.1f} dB" if interference > 0 else "    0    "
            lines.append(
                f"   {cp_length:3d}   {cp_length * SAMPLE_PERIOD * 1e6:5.3f}   "
                f"{100 * cp_length / block_length:6.2f} %   "
                f"{db(block_length / NUM_SUBCARRIERS):6.2f} dB     {interference_text}   "
                f"{sinr_db:5.1f} dB   {db(SNR) - sinr_db:5.1f} dB")
    return lines + [""]


def monte_carlo_summary(simulated):
    """Monte Carlo results next to the formula, as text lines."""
    lines = [f"Monte Carlo check: {NUM_CHANNEL_REALIZATIONS} random channel realizations "
             f"(seed {CP_CHECK_SEED}), no noise, OFDM symbols 1-13",
             "   Ncp   ISI+ICI formula   simulated   difference   SINR formula   simulated"]
    for scenario, profile in CHANNEL_PROFILES.items():
        delays, powers = profile["tap_delays"], profile["tap_powers"]
        measured_interference, measured_signal = simulated[scenario]
        lines.append(f"Scenario {scenario}")
        for index, cp_length in enumerate(SIMULATED_CP_LENGTHS[scenario]):
            formula = average_interference_power(delays, powers, cp_length, NUM_SUBCARRIERS)
            measured = measured_interference[index]
            sinr_formula = db(average_sinr(delays, powers, cp_length, NUM_SUBCARRIERS))
            sinr_measured = db(measured_signal[index] / (measured + 1 / SNR))
            if formula > 0:
                difference = f"{100 * (measured / formula - 1):+6.1f} %"
                lines.append(f"   {cp_length:3d}   {formula:.3e}       {measured:.3e}   "
                             f"{difference}     {sinr_formula:5.2f} dB     {sinr_measured:5.2f} dB")
            else:
                lines.append(f"   {cp_length:3d}   0               {measured:.1e}     "
                             f"  exact      {sinr_formula:5.2f} dB     {sinr_measured:5.2f} dB")
    return lines + [""]


def plot_interference_and_sinr(simulated):
    """Panel (a): average ISI + ICI power. Panel (b): average SINR. Both against Ncp."""
    noise_power_db = db(1 / SNR)
    fig, (ax_power, ax_sinr) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 2]},
                                            layout="constrained")
    fig.suptitle("Task 1: effect of the CP length (average over the channel)")

    for scenario, color, marker in [(1, BLUE, "o"), (2, ORANGE, "s")]:
        profile = CHANNEL_PROFILES[scenario]
        delays, powers = profile["tap_delays"], profile["tap_powers"]
        cp_design = minimum_cp_length(delays)

        # Formula: for Ncp >= L - 1 the interference is exactly zero (minus infinity in dB)
        cp_lengths = np.arange(cp_design)
        interference = [average_interference_power(delays, powers, cp, NUM_SUBCARRIERS)
                        for cp in cp_lengths]
        ax_power.plot(cp_lengths, db(interference), color=color, linewidth=1.4,
                      label=f"Scenario {scenario}: formula")
        all_cp_lengths = np.arange(86)
        sinr = [db(average_sinr(delays, powers, cp, NUM_SUBCARRIERS)) for cp in all_cp_lengths]
        ax_sinr.plot(all_cp_lengths, sinr, color=color, linewidth=1.4,
                     label=f"Scenario {scenario}: formula")

        # Monte Carlo points (zero interference cannot be drawn in dB)
        simulated_cp = np.array(SIMULATED_CP_LENGTHS[scenario])
        measured_interference, measured_signal = simulated[scenario]
        nonzero = simulated_cp < cp_design
        ax_power.plot(simulated_cp[nonzero], db(measured_interference[nonzero]),
                      linestyle="none", marker=marker, markersize=6, markerfacecolor="none",
                      markeredgewidth=1.3, color=color, label=f"Scenario {scenario}: simulation")
        ax_sinr.plot(simulated_cp, db(measured_signal / (measured_interference + 1 / SNR)),
                     linestyle="none", marker=marker, markersize=6, markerfacecolor="none",
                     markeredgewidth=1.3, color=color, label=f"Scenario {scenario}: simulation")

        for ax in (ax_power, ax_sinr):
            ax.axvline(cp_design, color=color, linestyle="--", linewidth=1.1)
        label_x, alignment = ((cp_design + 1, "left") if scenario == 1
                              else (cp_design - 1, "right"))
        ax_power.text(label_x, -54.3, f"$N_{{cp}} = {cp_design}$\n(no ISI from here on)",
                      color=color, fontsize=9, va="bottom", ha=alignment)

    ax_power.axhline(noise_power_db, color=INK, linestyle="-.", linewidth=1.1,
                     label="noise power $1/\\mathrm{SNR}$ ($-30$ dB)")
    ax_sinr.axhline(db(SNR), color=INK, linestyle="-.", linewidth=1.1,
                    label="SNR = 30 dB (no ISI)")
    for ax in (ax_power, ax_sinr):
        ax.axvline(NR_NORMAL_CP_LENGTH, color=GREY, linestyle=":", linewidth=1.3,
                   label="5G NR normal CP (36 samples)" if ax is ax_power else None)
        ax.grid(True)

    ax_power.set_ylim(-55, -10)
    ax_power.set_ylabel("average ISI + ICI power (dB)")
    ax_power.set_title("(a) Average ISI + ICI power per subcarrier")
    ax_power.legend(loc="upper right", fontsize=8, ncol=2)

    ax_sinr.set_xlim(-1, 85)
    ax_sinr.set_ylim(12, 32)
    ax_sinr.set_xlabel("CP length $N_{cp}$ (samples)")
    ax_sinr.set_ylabel("average SINR (dB)")
    ax_sinr.set_title("(b) Average SINR per subcarrier")
    ax_sinr.legend(loc="lower right", fontsize=8, ncol=2)
    save_figure(fig, RESULTS_FOLDER, "interference_vs_cp")


if __name__ == "__main__":
    rng = np.random.default_rng(CP_CHECK_SEED)
    simulated = {scenario: simulate_average_interference(
                     scenario, SIMULATED_CP_LENGTHS[scenario], NUM_CHANNEL_REALIZATIONS, rng)
                 for scenario in CHANNEL_PROFILES}

    summary = design_summary() + candidate_summary() + monte_carlo_summary(simulated)
    print("\n".join(summary))

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    (RESULTS_FOLDER / "cp_design.txt").write_text("\n".join(summary) + "\n")
    plot_interference_and_sinr(simulated)
    print(f"Saved the summary and the figure to {RESULTS_FOLDER}")
