"""Extra: the link of main.py for every CP length from 0 to 80, and a check of the formula for D[k]."""
import numpy as np
import matplotlib.pyplot as plt

from main import RESULTS_FOLDER, analyse, run_link     # also sets up the import path
from parameters import (CHANNEL_PROFILES, CHANNEL_SEEDS, NUM_DATA_SYMBOLS,
                        NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, SHORT_CP_LENGTH, SNR)
from cp_design import interference_power_per_subcarrier, minimum_cp_length
from transmitter import generate_data, ofdm_transmitter
from channel import channel_taps, frequency_response, multipath_channel
from receiver import ofdm_receiver
from plot_style import BLUE, INK, ORANGE, save_figure

SWEEP_FOLDER = RESULTS_FOLDER / "cp_sweep"
CP_LENGTHS = np.arange(0, 81)
TABLE_CP_LENGTHS = [0, 4, 10, 20, 36, 55, 69, 74, 75, 80]
NUM_CHECK_SLOTS = 200


def sweep(scenario):
    """Return the measured and expected disturbance power and symbol errors for every Ncp."""
    measured_power, expected_power = [], []
    measured_errors, expected_errors = [], []
    for cp_length in CP_LENGTHS:
        numbers = analyse(run_link(scenario, cp_length), cp_length)
        measured_power.append(numbers["disturbance_power"])
        expected_power.append(numbers["predicted_disturbance"])
        measured_errors.append(numbers["errors_slot"])
        expected_errors.append(numbers["expected_errors_slot"])
    return (np.array(measured_power), np.array(expected_power),
            np.array(measured_errors), np.array(expected_errors))


def check_interference_formula(scenario, cp_length):
    """Compare the ISI + ICI power of many noise-free slots of random data with D[k]."""
    profile = CHANNEL_PROFILES[scenario]
    taps = channel_taps(profile["tap_delays"], profile["tap_powers"],
                        np.random.default_rng(CHANNEL_SEEDS[scenario]))
    response = frequency_response(taps, NUM_SUBCARRIERS)
    expected = interference_power_per_subcarrier(taps, cp_length, NUM_SUBCARRIERS)

    power_per_slot, power_first_symbol = [], []
    power_per_subcarrier = np.zeros(NUM_SUBCARRIERS)
    for slot in range(NUM_CHECK_SLOTS):
        data = generate_data(NUM_DATA_SYMBOLS, np.random.default_rng(slot))
        x, grid = ofdm_transmitter(data, NUM_SUBCARRIERS, cp_length)
        y = multipath_channel(x, taps, np.inf, np.random.default_rng(0))   # no noise
        _, received_grid, _ = ofdm_receiver(y, response, NUM_SUBCARRIERS, cp_length,
                                            NUM_OFDM_SYMBOLS)
        error_power = np.abs(received_grid - response[:, np.newaxis] * grid) ** 2
        interference = error_power[:, 1:]                 # symbols 1-13
        power_first_symbol.append(np.mean(error_power[:, 0]))
        power_per_slot.append(np.mean(interference))
        power_per_subcarrier += np.mean(interference, axis=1) / NUM_CHECK_SLOTS

    ratio = power_per_subcarrier / expected
    return [
        f"Check of the ISI + ICI formula: Scenario {scenario}, Ncp = {cp_length}, "
        f"no noise, {NUM_CHECK_SLOTS} slots of random data (symbols 1-13)",
        f"  expected mean of D[k]                  : {np.mean(expected):.4e}",
        f"  measured, mean over all slots          : {np.mean(power_per_slot):.4e} "
        f"({100 * (np.mean(power_per_slot) / np.mean(expected) - 1):+.2f} %)",
        f"  spread of one slot (std / mean)        : "
        f"{100 * np.std(power_per_slot) / np.mean(power_per_slot):.1f} %",
        f"  per subcarrier: measured / D[k], mean  : {np.mean(ratio):.4f}",
        f"  per subcarrier: measured / D[k], std   : {np.std(ratio):.4f}",
        f"  symbol 0 (nothing before it) / mean D  : "
        f"{np.mean(power_first_symbol) / np.mean(expected):.3f}",
    ]


def plot_sweep(results):
    """Plot the disturbance power (top) and the symbol errors (bottom) against the CP length."""
    fig, (ax_power, ax_errors) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                              gridspec_kw={"height_ratios": [3, 2]})
    for scenario, color, marker in [(1, BLUE, "o"), (2, ORANGE, "s")]:
        measured_power, expected_power, measured_errors, expected_errors = results[scenario]
        cp_design = minimum_cp_length(CHANNEL_PROFILES[scenario]["tap_delays"])

        ax_power.plot(CP_LENGTHS, 10 * np.log10(expected_power), color=color, linewidth=1.3,
                      label=f"Scenario {scenario}: expected $1/\\mathrm{{SNR}} + D$")
        ax_power.plot(CP_LENGTHS, 10 * np.log10(measured_power), linestyle="none",
                      marker=marker, markersize=3.5, markerfacecolor="none", color=color,
                      label=f"Scenario {scenario}: measured")
        ax_errors.plot(CP_LENGTHS, expected_errors, color=color, linewidth=1.3,
                       label=f"Scenario {scenario}: expected")
        ax_errors.plot(CP_LENGTHS, measured_errors, linestyle="none", marker=marker,
                       markersize=3.5, markerfacecolor="none", color=color,
                       label=f"Scenario {scenario}: measured")
        for ax in (ax_power, ax_errors):
            ax.axvline(cp_design, color=color, linestyle="--", linewidth=1.0)

    ax_power.axhline(10 * np.log10(1 / SNR), color=INK, linestyle="-.", linewidth=1.0,
                     label="noise power $1/\\mathrm{SNR}$")
    ax_power.set_ylabel("power of $\\tilde{y}_i[k] - H[k]\\tilde{d}_i[k]$ (dB)")
    ax_power.set_title("Both scenarios with every CP length from 0 to 80 samples")
    ax_power.grid(True)
    ax_power.legend(loc="upper right", fontsize=8, ncol=2)

    ax_errors.set_xlabel("CP length $N_{cp}$ (samples)")
    ax_errors.set_ylabel("symbol errors in the slot")
    ax_errors.set_xlim(CP_LENGTHS[0] - 1, CP_LENGTHS[-1] + 1)
    ax_errors.grid(True)
    ax_errors.legend(loc="upper right", fontsize=8, ncol=2)
    save_figure(fig, SWEEP_FOLDER, "cp_sweep")


if __name__ == "__main__":
    results = {scenario: sweep(scenario) for scenario in CHANNEL_PROFILES}

    lines = ["CP sweep: disturbance power over symbols 1-13 (dB), symbol errors in the slot", ""]
    for scenario, (measured_power, expected_power, measured_errors,
                   expected_errors) in results.items():
        lines.append(f"Scenario {scenario}")
        lines.append("   Ncp   measured   expected   errors   expected errors")
        for cp_length in TABLE_CP_LENGTHS:
            lines.append(f"   {cp_length:3d}   {10 * np.log10(measured_power[cp_length]):7.2f}"
                         f"    {10 * np.log10(expected_power[cp_length]):7.2f}"
                         f"    {measured_errors[cp_length]:5d}"
                         f"    {expected_errors[cp_length]:8.2f}")
        lines.append("")
    lines += check_interference_formula(scenario=2, cp_length=SHORT_CP_LENGTH)

    print("\n".join(lines))
    SWEEP_FOLDER.mkdir(parents=True, exist_ok=True)
    (SWEEP_FOLDER / "summary.txt").write_text("\n".join(lines) + "\n")
    plot_sweep(results)
    print(f"Saved the figure and summary to {SWEEP_FOLDER}")
