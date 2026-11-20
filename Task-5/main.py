"""Task 5 main script: python3 Task-5/main.py --scenario S [--cp NCP] makes all figures of one run."""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import erfc

# The blocks of the link are in the folders of Tasks 1-4
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for folder in ["", "Task-1", "Task-2", "Task-3", "Task-4"]:
    sys.path.append(str(PROJECT_ROOT / folder))

from parameters import (CHANNEL_PROFILES, CHANNEL_SEEDS, DATA_SEED, NOISE_SEED,
                        NUM_DATA_SYMBOLS, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, PLOT_SYMBOL,
                        SAMPLE_PERIOD, SNR)
from cp_design import interference_power_per_subcarrier, minimum_cp_length
from transmitter import generate_data, ofdm_transmitter, qam4_constellation
from channel import channel_taps, frequency_response, multipath_channel
from receiver import detect_symbols, ofdm_receiver
import plots

RESULTS_FOLDER = PROJECT_ROOT / "results" / "Task-5"

# y-axis (dB) of the equalization error plot; both Scenario 2 runs share it for comparison
ERROR_PLOT_LIMITS = {1: (-60, -10), 2: (-60, 15)}


@dataclass
class LinkSignals:
    """All signals of one slot, from the data s to the detected symbols."""
    data: np.ndarray             # s, 7168 symbols
    grid: np.ndarray             # transmitted resource grid D~, N x 14
    x: np.ndarray                # transmitted signal, M samples
    taps: np.ndarray             # channel taps h, L values
    response: np.ndarray         # channel frequency response H[k], N values
    y: np.ndarray                # received signal, M + L - 1 samples
    received_grid: np.ndarray    # y~_i[k] after the DFT, N x 14
    equalized_grid: np.ndarray   # d^_i[k] after the equalizer, N x 14
    estimate: np.ndarray         # s^, 7168 values
    detected: np.ndarray         # closest 4-QAM point to each value of s^


def run_link(scenario, cp_length):
    """Send one slot through transmitter, channel and receiver."""
    profile = CHANNEL_PROFILES[scenario]

    # Task 2: transmitter
    data = generate_data(NUM_DATA_SYMBOLS, np.random.default_rng(DATA_SEED))
    x, grid = ofdm_transmitter(data, NUM_SUBCARRIERS, cp_length)

    # Task 3: channel, one realization with a fixed seed per scenario
    taps = channel_taps(profile["tap_delays"], profile["tap_powers"],
                        np.random.default_rng(CHANNEL_SEEDS[scenario]))
    response = frequency_response(taps, NUM_SUBCARRIERS)
    y = multipath_channel(x, taps, SNR, np.random.default_rng(NOISE_SEED))

    # Task 4: receiver, then a 4-QAM decision to count symbol errors
    estimate, received_grid, equalized_grid = ofdm_receiver(
        y, response, NUM_SUBCARRIERS, cp_length, NUM_OFDM_SYMBOLS)
    detected = detect_symbols(estimate, qam4_constellation())

    return LinkSignals(data, grid, x, taps, response, y, received_grid,
                       equalized_grid, estimate, detected)


# Numbers for the report

def qfunc(value):
    """Return the Gaussian tail function Q(x) = 0.5 erfc(x / sqrt(2))."""
    return 0.5 * erfc(value / np.sqrt(2))


def qam4_error_probability(snr):
    """Return the 4-QAM symbol error probability 2 Q(sqrt(snr)) - Q(sqrt(snr))^2."""
    q = qfunc(np.sqrt(snr))
    return 2 * q - q ** 2


def analyse(signals, cp_length):
    """Return every number quoted in the report for one run."""
    noise_power = 1 / SNR
    channel_power = np.abs(signals.response) ** 2
    # Averages use symbols 1-13, because symbol 0 sees only half the ISI + ICI
    later_symbols = slice(1, NUM_OFDM_SYMBOLS)

    # ISI + ICI power per subcarrier predicted from the channel (zero if Ncp >= L - 1)
    interference = interference_power_per_subcarrier(signals.taps, cp_length, NUM_SUBCARRIERS)

    # Disturbance before the equalizer: y~ - H d~ = noise (+ ISI and ICI)
    disturbance = signals.received_grid - signals.response[:, np.newaxis] * signals.grid
    # Error after the equalizer: d^ - d~ = disturbance / H
    error_power = np.abs(signals.equalized_grid - signals.grid) ** 2
    predicted_error = (noise_power + interference) / channel_power

    # Symbol errors per OFDM symbol, and the expected number from the SINR of every subcarrier
    errors = (signals.detected != signals.data).reshape(NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS)
    sinr = channel_power / (noise_power + interference)
    sinr_first_symbol = channel_power / (noise_power + interference / 2)
    expected_errors = (np.sum(qam4_error_probability(sinr_first_symbol))
                       + (NUM_OFDM_SYMBOLS - 1) * np.sum(qam4_error_probability(sinr)))

    x_power = np.abs(signals.x) ** 2
    strong = channel_power >= 1                    # subcarriers with |H[k]|^2 >= 0 dB
    error_subcarriers = np.unique(np.nonzero(errors)[1])
    return {
        # arrays used by the figures
        "interference": interference,
        "error_power": error_power,
        "predicted_error": predicted_error,
        # (a) channel
        "channel_energy": np.sum(np.abs(signals.taps) ** 2),
        "channel_min_db": 10 * np.log10(channel_power.min()),
        "channel_min_k": int(np.argmin(channel_power)),
        "channel_max_db": 10 * np.log10(channel_power.max()),
        "channel_max_k": int(np.argmax(channel_power)),
        "weak_subcarriers": int(np.sum(channel_power < 0.1)),     # below -10 dB
        # (b) transmitted signal
        "x_power": np.mean(x_power),
        "x_mean_amplitude": np.mean(np.abs(signals.x)),
        "papr_db": 10 * np.log10(x_power.max() / x_power.mean()),
        # (c) disturbance before the equalizer and SINR
        "disturbance_power": np.mean(np.abs(disturbance[:, later_symbols]) ** 2),
        "predicted_disturbance": noise_power + np.mean(interference),
        "interference_mean": np.mean(interference),
        "mean_sinr_db": 10 * np.log10(np.mean(channel_power)
                                      / (noise_power + np.mean(interference))),
        "sinr_min_db": 10 * np.log10(sinr.min()),
        "sinr_max_db": 10 * np.log10(sinr.max()),
        # (c) error after the equalizer
        "error_plot_symbol": np.mean(error_power[:, PLOT_SYMBOL]),
        "error_slot": np.mean(error_power[:, later_symbols]),
        "predicted_error_mean": np.mean(predicted_error),
        "strong_subcarriers": int(strong.sum()),
        "error_strong": np.mean(error_power[strong][:, later_symbols]),
        "predicted_error_strong": np.mean(predicted_error[strong]),
        # symbol errors after the 4-QAM decision
        "errors_plot_symbol": int(errors[PLOT_SYMBOL].sum()),
        "errors_slot": int(errors.sum()),
        "expected_errors_plot_symbol": np.sum(qam4_error_probability(sinr)),
        "expected_errors_slot": expected_errors,
        "error_subcarriers": error_subcarriers,
        "error_subcarrier_power_db": 10 * np.log10(channel_power[error_subcarriers]),
    }


def write_summary(scenario, cp_length, signals, numbers, folder):
    """Print the numbers of one run and save them to summary.txt."""
    n = numbers
    block_length = NUM_SUBCARRIERS + cp_length
    lines = [
        f"Scenario {scenario}, Ncp = {cp_length} samples = "
        f"{cp_length * SAMPLE_PERIOD * 1e6:.3f} us "
        f"(L - 1 = {len(signals.taps) - 1}), SNR = {10 * np.log10(SNR):.0f} dB",
        f"seeds: data {DATA_SEED}, channel {CHANNEL_SEEDS[scenario]}, noise {NOISE_SEED}",
        "",
        "Signal lengths",
        f"  M = 14 (N + Ncp)                    : {len(signals.x)}",
        f"  length of y, M + L - 1              : {len(signals.y)}",
        f"  CP overhead Ncp / (N + Ncp)         : {100 * cp_length / block_length:.2f} %",
        "",
        "(a) Channel",
        f"  ||h||^2 = mean of |H[k]|^2           : {n['channel_energy']:.4f} "
        f"({10 * np.log10(n['channel_energy']):.2f} dB)",
        f"  |H[k]|^2 min                        : {n['channel_min_db']:.2f} dB at k = {n['channel_min_k']}",
        f"  |H[k]|^2 max                        : {n['channel_max_db']:.2f} dB at k = {n['channel_max_k']}",
        f"  subcarriers with |H[k]|^2 < -10 dB  : {n['weak_subcarriers']}",
        "",
        "(b) Transmitted signal",
        f"  mean |x[m]|^2                       : {n['x_power']:.4f}",
        f"  mean |x[m]| (Rayleigh: 0.886)       : {n['x_mean_amplitude']:.4f}",
        f"  PAPR = max |x|^2 / mean |x|^2       : {n['papr_db']:.2f} dB",
        "",
        "(c) Equalization",
        "  Disturbance y~ - H d~ (symbols 1-13, all subcarriers):",
        f"    measured power                    : {n['disturbance_power']:.3e} "
        f"({10 * np.log10(n['disturbance_power']):.2f} dB)",
        f"    expected 1/SNR + mean D[k]        : {n['predicted_disturbance']:.3e} "
        f"({10 * np.log10(n['predicted_disturbance']):.2f} dB)",
        f"    of which ISI + ICI, mean D[k]     : {n['interference_mean']:.3e}",
        f"    mean SINR = ||h||^2 / (1/SNR + D) : {n['mean_sinr_db']:.2f} dB",
        "  Equalization error |d^ - d~|^2:",
        f"    mean over symbol {PLOT_SYMBOL}               : {n['error_plot_symbol']:.3e}",
        f"    mean over symbols 1-13            : {n['error_slot']:.3e}",
        f"    expected mean_k (1/SNR + D)/|H|^2 : {n['predicted_error_mean']:.3e}",
        f"    on the {n['strong_subcarriers']} subcarriers with |H|^2 >= 0 dB: "
        f"{n['error_strong']:.3e} (expected {n['predicted_error_strong']:.3e})",
        f"  SINR per subcarrier |H|^2 / (1/SNR + D[k]): "
        f"{n['sinr_min_db']:.2f} dB to {n['sinr_max_db']:.2f} dB",
        "  Symbol errors after a 4-QAM decision:",
        f"    in symbol {PLOT_SYMBOL} (512 symbols)        : {n['errors_plot_symbol']} "
        f"(expected {n['expected_errors_plot_symbol']:.2f})",
        f"    in the slot (7168 symbols)        : {n['errors_slot']} "
        f"(expected {n['expected_errors_slot']:.2f})",
        f"    on {len(n['error_subcarriers'])} different subcarriers k: "
        + (", ".join(f"{k} ({p:.1f} dB)" for k, p in
                     zip(n['error_subcarriers'], n['error_subcarrier_power_db'])) or "none"),
    ]
    # Share of each late tap in the ISI + ICI power, (2/N) |h[l]|^2 (l - Ncp)
    late_taps = [l for l in range(cp_length + 1, len(signals.taps)) if signals.taps[l] != 0]
    if late_taps:
        tap_terms = {l: 2 / NUM_SUBCARRIERS * abs(signals.taps[l]) ** 2 * (l - cp_length)
                     for l in late_taps}
        lines += ["", "  ISI + ICI from each tap with delay l > Ncp:",
                  "     l   |h[l]|^2   l - Ncp   power      share"]
        for l, power in tap_terms.items():
            lines.append(f"    {l:2d}   {abs(signals.taps[l]) ** 2:.4f}   {l - cp_length:7d}"
                         f"   {power:.2e}   {100 * power / sum(tap_terms.values()):5.1f} %")

        # How much larger the expected error is than with noise only
        interference = n["interference"]
        increase_db = 10 * np.log10(1 + interference * SNR)
        lines += [
            f"  D[k] over the subcarriers           : {10 * np.log10(interference.min()):.1f} dB"
            f" to {10 * np.log10(interference.max()):.1f} dB",
            f"  expected error / noise-only error   : {increase_db.min():.1f} dB to "
            f"{increase_db.max():.1f} dB (median {np.median(increase_db):.1f} dB)",
        ]

    print("\n".join(lines))
    (folder / "summary.txt").write_text("\n".join(lines) + "\n")


def make_figures(scenario, cp_length, signals, numbers, folder):
    """Make all figures of Task 5 (a)-(c) for one run."""
    profile = CHANNEL_PROFILES[scenario]
    name = f"Scenario {scenario}, $N_{{cp}} = {cp_length}$"
    i = PLOT_SYMBOL

    # Same colour scale for both scatter plots of a run
    power_db = 10 * np.log10(np.abs(signals.response) ** 2)
    color_range = (np.floor(power_db.min()), np.ceil(power_db.max()))

    # (a) channel
    plots.plot_channel_taps(signals.taps, profile["tap_delays"], profile["tap_powers"],
                            f"Scenario {scenario}: channel taps", folder)
    plots.plot_frequency_response(signals.response,
                                  f"Scenario {scenario}: frequency response", folder)
    # (b) transmitted signal
    plots.plot_transmitted_signal(signals.x, NUM_SUBCARRIERS, cp_length,
                                  f"{name}: transmitted signal of one slot", folder)
    plots.plot_amplitude_histogram(signals.x, f"{name}: amplitude of $x[m]$", folder)
    # (c) equalization
    plots.plot_received_symbols(signals.received_grid[:, i], signals.response,
                                f"{name}: received $\\tilde{{y}}_{{{i}}}[k]$",
                                folder, color_range)
    plots.plot_equalized_symbols(signals.equalized_grid[:, i], signals.response,
                                 qam4_constellation(),
                                 f"{name}: equalized $\\hat{{\\tilde{{d}}}}_{{{i}}}[k]$",
                                 folder, color_range)
    plots.plot_equalization_error(numbers["error_power"][:, i],
                                  np.mean(numbers["error_power"][:, 1:], axis=1),
                                  numbers["predicted_error"], i,
                                  f"{name}: equalization error on every subcarrier",
                                  folder, ERROR_PLOT_LIMITS[scenario])


def main(scenario, cp_length=None):
    """Run the link for one scenario and CP length; save figures and summary."""
    if cp_length is None:
        cp_length = minimum_cp_length(CHANNEL_PROFILES[scenario]["tap_delays"])

    folder = RESULTS_FOLDER / f"scenario{scenario}_cp{cp_length}"
    folder.mkdir(parents=True, exist_ok=True)

    signals = run_link(scenario, cp_length)
    numbers = analyse(signals, cp_length)
    write_summary(scenario, cp_length, signals, numbers, folder)
    make_figures(scenario, cp_length, signals, numbers, folder)
    print(f"Saved figures and summary to {folder}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OFDM link of Assignment 3, Task 5")
    parser.add_argument("--scenario", type=int, choices=[1, 2], required=True,
                        help="channel scenario of Table 1")
    parser.add_argument("--cp", type=int, default=None,
                        help="CP length in samples (default: the Task 1 design, L - 1)")
    arguments = parser.parse_args()
    main(arguments.scenario, arguments.cp)
