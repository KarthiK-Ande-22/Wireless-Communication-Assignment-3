"""Task 1: CP length design for both scenarios. Run this file to make Table 1, Table 2 and Fig. 1."""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# parameters and plot style are in the project root, the link blocks in Tasks 2-4
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


# Design functions

def minimum_cp_length(tap_delays):
    """Return the shortest CP without ISI, L - 1 (delay of the last tap)."""
    return max(tap_delays)


def rms_delay_spread(tap_delays, tap_powers):
    """Return the rms delay spread in samples."""
    delays = np.asarray(tap_delays, dtype=float)
    weights = np.asarray(tap_powers) / np.sum(tap_powers)
    mean_delay = np.sum(weights * delays)
    return np.sqrt(np.sum(weights * (delays - mean_delay) ** 2))


def average_interference_power(tap_delays, tap_powers, cp_length, num_subcarriers):
    """Return P_int = (2 / N) sum_l sigma_l^2 max(l - Ncp, 0), the average ISI + ICI power."""
    excess_delays = np.maximum(np.asarray(tap_delays) - cp_length, 0)
    return 2 / num_subcarriers * np.sum(np.asarray(tap_powers) * excess_delays)


def average_sinr(tap_delays, tap_powers, cp_length, num_subcarriers):
    """Return the average SINR per subcarrier, E|H[k]|^2 / (P_int + 1/SNR)."""
    signal_power = np.sum(tap_powers)
    interference = average_interference_power(tap_delays, tap_powers, cp_length,
                                              num_subcarriers)
    return signal_power / (interference + 1 / SNR)


def interference_power_per_subcarrier(taps, cp_length, num_subcarriers):
    """Return D[k], the ISI + ICI power on each subcarrier for one channel (symbols 1-13)."""
    power = np.zeros(num_subcarriers)
    for first_late_tap in range(cp_length + 1, len(taps)):
        late_taps = np.zeros_like(taps)
        late_taps[first_late_tap:] = taps[first_late_tap:]
        power += np.abs(np.fft.fft(late_taps, num_subcarriers)) ** 2
    return 2 / num_subcarriers * power


def shortest_cp_below(tap_delays, tap_powers, power_limit, num_subcarriers):
    """Return the smallest Ncp whose average ISI + ICI power is at most power_limit."""
    cp_length = 0
    while average_interference_power(tap_delays, tap_powers, cp_length,
                                     num_subcarriers) > power_limit:
        cp_length += 1
    return cp_length


def db(value):
    """Return 10 log10(value)."""
    return 10 * np.log10(value)


# Monte Carlo check of P_int with the link of Tasks 2-4

def simulate_average_interference(scenario, cp_lengths, num_realizations, rng):
    """Return the measured ISI + ICI and signal power over many noise-free channels (symbols 1-13)."""
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


# Report tables and figure

def design_summary():
    """Return the CP design of both scenarios as text lines."""
    noise_power = 1 / SNR
    lines = ["Task 1: CP design (N = 512, W = 15.36 MHz, sample period 65.1 ns)",
             "The same Ncp is used for all 14 OFDM symbols of a slot.", ""]

    for scenario, profile in CHANNEL_PROFILES.items():
        delays, powers = profile["tap_delays"], profile["tap_powers"]
        cp_length = minimum_cp_length(delays)
        delay_spread = rms_delay_spread(delays, powers)
        # Rule of thumb: B_c = 1 / (5 tau_rms)
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
    """Return the cost and SINR of the candidate CP lengths as text lines."""
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
    """Return the Monte Carlo results next to the formula as text lines."""
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
    """Plot (a) the average ISI + ICI power and (b) the average SINR against Ncp."""
    noise_power_db = db(1 / SNR)
    fig, (ax_power, ax_sinr) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 2]},
                                            layout="constrained")
    fig.suptitle("Task 1: effect of the CP length (average over the channel)")

    for scenario, color, marker in [(1, BLUE, "o"), (2, ORANGE, "s")]:
        profile = CHANNEL_PROFILES[scenario]
        delays, powers = profile["tap_delays"], profile["tap_powers"]
        cp_design = minimum_cp_length(delays)

        # Formula; zero for Ncp >= L - 1, which cannot be drawn in dB
        cp_lengths = np.arange(cp_design)
        interference = [average_interference_power(delays, powers, cp, NUM_SUBCARRIERS)
                        for cp in cp_lengths]
        ax_power.plot(cp_lengths, db(interference), color=color, linewidth=1.4,
                      label=f"Scenario {scenario}: formula")
        all_cp_lengths = np.arange(86)
        sinr = [db(average_sinr(delays, powers, cp, NUM_SUBCARRIERS)) for cp in all_cp_lengths]
        ax_sinr.plot(all_cp_lengths, sinr, color=color, linewidth=1.4,
                     label=f"Scenario {scenario}: formula")

        # Monte Carlo points
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
