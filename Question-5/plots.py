"""Figures of Task 5. Every function makes one figure and saves it as PDF and PNG.

All plots use the sample, tap or subcarrier index on the horizontal axis,
except the scatter plots, which show values in the complex plane.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))   # for plot_style.py
from plot_style import BLUE, GREY, INK, ORANGE, save_figure

COLORMAP = "viridis"          # colour of the scatter points = |H[k]|^2 in dB
EQUALIZED_AXIS_LIMIT = 2.0    # same axes for all equalized scatter plots


def power_db(values):
    """10 log10 |values|^2."""
    return 10 * np.log10(np.abs(values) ** 2)


# ---------------------------------------------------------------------------
# (a) Channel
# ---------------------------------------------------------------------------
def plot_channel_taps(taps, tap_delays, tap_powers, title, folder):
    """Stem plot of |h[l]| against the tap index l.

    The grey bars show the rms value sigma_l of each tap (Table 1), to compare
    this realization with the average power profile.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    delays = np.asarray(tap_delays)
    markers, stems, baseline = ax.stem(delays, np.abs(taps[delays]), basefmt=" ",
                                       label="$|h[\\ell]|$ (this realization)")
    plt.setp(markers, color=BLUE, markersize=5)
    plt.setp(stems, color=BLUE, linewidth=1.4)
    ax.plot(delays, np.sqrt(tap_powers), linestyle="none", marker="_", markersize=12,
            markeredgewidth=2, color=ORANGE, label="rms value $\\sigma_\\ell$ (Table 1)")
    ax.axhline(0, color=INK, linewidth=0.8)

    ax.set_xlim(-0.04 * len(taps) - 0.5, len(taps) - 0.5 + 0.04 * len(taps))
    ax.set_ylim(0, 1.15 * max(np.max(np.abs(taps)), np.max(np.sqrt(tap_powers))))
    ax.set_xlabel("tap index $\\ell$")
    ax.set_ylabel("$|h[\\ell]|$")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper right")
    save_figure(fig, folder, "channel_taps")


def plot_frequency_response(response, title, folder):
    """|H[k]|^2 in dB against the subcarrier index k."""
    num_subcarriers = len(response)
    mean_power_db = 10 * np.log10(np.mean(np.abs(response) ** 2))

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(np.arange(num_subcarriers), power_db(response), color=BLUE, linewidth=1.2,
            label="$|H[k]|^2$")
    ax.axhline(mean_power_db, color=ORANGE, linestyle="--", linewidth=1.2,
               label=f"mean over $k$ = $\\|\\mathbf{{h}}\\|^2$ (${mean_power_db:.2f}$ dB)")
    lowest, highest = power_db(response).min(), power_db(response).max()
    ax.set_ylim(lowest - 1, highest + 0.4 * (highest - lowest))   # room for the legend
    ax.set_xlim(0, num_subcarriers - 1)
    ax.set_xlabel("subcarrier index $k$")
    ax.set_ylabel("$|H[k]|^2$ (dB)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    save_figure(fig, folder, "frequency_response")


# ---------------------------------------------------------------------------
# (b) Transmitted signal
# ---------------------------------------------------------------------------
def plot_transmitted_signal(x, num_subcarriers, cp_length, title, folder):
    """|x[m]| against the sample index m, with the CP of every OFDM symbol shaded.

    A side panel shows the histogram of |x[m]| next to the Rayleigh density
    2 r exp(-r^2), the amplitude of a CN(0, 1) sample.
    """
    block_length = num_subcarriers + cp_length
    num_ofdm_symbols = len(x) // block_length
    amplitude = np.abs(x)

    fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True,
                                      gridspec_kw={"width_ratios": [5, 1], "wspace": 0.04})
    for i in range(num_ofdm_symbols):
        start = i * block_length
        ax.axvspan(start, start + cp_length, color=ORANGE, alpha=0.35, linewidth=0,
                   label="cyclic prefix" if i == 0 else None)
        ax.axvline(start, color=GREY, linewidth=0.5, linestyle=":")
    ax.plot(np.arange(len(x)), amplitude, color=BLUE, linewidth=0.35, label="$|x[m]|$")
    ax.axhline(1, color=INK, linestyle="--", linewidth=1.0,
               label="rms value 1 ($E|x[m]|^2 = 1$)")
    ax.set_xlim(0, len(x))
    ax.set_ylim(0, 1.3 * amplitude.max())     # room for the legend
    ax.set_xlabel("sample index $m$")
    ax.set_ylabel("$|x[m]|$")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=3, framealpha=0.95)

    # Histogram of the amplitudes, drawn sideways so it shares the |x| axis
    bins = np.linspace(0, amplitude.max(), 40)
    ax_hist.hist(amplitude, bins=bins, density=True, orientation="horizontal",
                 color=BLUE, alpha=0.45)
    r = np.linspace(0, amplitude.max(), 200)
    ax_hist.plot(2 * r * np.exp(-r ** 2), r, color=INK, linewidth=1.2, label="Rayleigh pdf")
    ax_hist.set_xlabel("density")
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.tick_params(labelleft=False)
    save_figure(fig, folder, "transmitted_signal")


# ---------------------------------------------------------------------------
# (c) Equalization
# ---------------------------------------------------------------------------
def scatter_colored_by_channel(ax, values, response, color_range):
    """Scatter plot of complex values, coloured by |H[k]|^2 in dB."""
    order = np.argsort(np.abs(response))[::-1]   # weakest subcarriers drawn last (on top)
    return ax.scatter(values[order].real, values[order].imag, c=power_db(response[order]),
                      cmap=COLORMAP, vmin=color_range[0], vmax=color_range[1],
                      s=9, linewidths=0)


def plot_received_symbols(received, response, title, folder, color_range):
    """Received values y~_i[k] (after the DFT, before the equalizer)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    points = scatter_colored_by_channel(ax, received, response, color_range)
    fig.colorbar(points, ax=ax, label="$|H[k]|^2$ (dB)", shrink=0.9)

    limit = 1.1 * np.max(np.abs(np.concatenate([received.real, received.imag])))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axhline(0, color=GREY, linewidth=0.6)
    ax.axvline(0, color=GREY, linewidth=0.6)
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.set_title(title)
    ax.grid(True)
    save_figure(fig, folder, "received_symbols")


def plot_equalized_symbols(equalized, response, constellation, title, folder, color_range):
    """Equalized values d^_i[k] together with the 4-QAM constellation points.

    The axes are the same in every run, so the plots can be compared. Points
    outside the axes are counted in the legend.
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    points = scatter_colored_by_channel(ax, equalized, response, color_range)
    fig.colorbar(points, ax=ax, label="$|H[k]|^2$ (dB)", shrink=0.9)
    ax.plot(constellation.real, constellation.imag, linestyle="none", marker="+",
            markersize=14, markeredgewidth=2.2, color="#d62728", label="4-QAM points")

    limit = EQUALIZED_AXIS_LIMIT
    outside = np.sum((np.abs(equalized.real) > limit) | (np.abs(equalized.imag) > limit))
    if outside > 0:
        ax.plot([], [], linestyle="none", label=f"{outside} point(s) outside the axes")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axhline(0, color=GREY, linewidth=0.6)
    ax.axvline(0, color=GREY, linewidth=0.6)
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    save_figure(fig, folder, "equalized_symbols")


def plot_equalization_error(error_one_symbol, error_average, predicted_error,
                            plot_symbol, title, folder, y_limits):
    """Error power |d^_i[k] - d~_i[k]|^2 in dB against the subcarrier index k.

    Grey dots: the plotted OFDM symbol. Blue: mean over the OFDM symbols
    1, ..., 13. Black: the expected error power (1/SNR + D[k]) / |H[k]|^2.
    """
    num_subcarriers = len(error_one_symbol)
    subcarriers = np.arange(num_subcarriers)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(subcarriers, 10 * np.log10(error_one_symbol), linestyle="none", marker=".",
            markersize=3, color=GREY, label=f"OFDM symbol $i = {plot_symbol}$")
    ax.plot(subcarriers, 10 * np.log10(error_average), color=BLUE, linewidth=1.0,
            label="mean over OFDM symbols $i = 1, \\ldots, 13$")
    ax.plot(subcarriers, 10 * np.log10(predicted_error), color=INK, linestyle="--",
            linewidth=1.2, label="expected: $(1/\\mathrm{SNR} + D[k])\\,/\\,|H[k]|^2$")
    ax.set_xlim(0, num_subcarriers - 1)
    ax.set_ylim(y_limits)
    ax.set_xlabel("subcarrier index $k$")
    ax.set_ylabel("$|\\hat{\\tilde{d}}_i[k] - \\tilde{d}_i[k]|^2$ (dB)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    save_figure(fig, folder, "equalization_error")
