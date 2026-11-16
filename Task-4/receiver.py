"""Task 4: OFDM receiver (Fig. 3 of the assignment). The receiver knows h.

    y --> split into 14 blocks --> remove CP --> DFT F --> equalize --> demap --> s^

Each block is a separate function. Running this file sends a slot through
the channel without noise and checks that the receiver gets the data back:

    python3 Task-4/receiver.py
"""
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# (a) OFDM demodulation: split, remove CP, DFT
# ---------------------------------------------------------------------------
def split_into_blocks(received, num_subcarriers, cp_length, num_ofdm_symbols):
    """Cut y into 14 consecutive blocks y_i of N + Ncp samples.

    Returns an (N + Ncp) x 14 array whose column i is y_i. The last L - 1
    samples of y (tail of the last OFDM symbol) are not used.
    """
    block_length = num_subcarriers + cp_length
    used_samples = received[:num_ofdm_symbols * block_length]
    return used_samples.reshape(num_ofdm_symbols, block_length).T


def remove_cyclic_prefix(blocks, cp_length):
    """Drop the first Ncp samples of every block: y_c,i has N samples."""
    return blocks[cp_length:, :]


def ofdm_demodulate(time_blocks):
    """y~_i = F y_c,i for every column, with the unitary DFT matrix F.

    numpy's fft has no factor 1/sqrt(N), so fft / sqrt(N) is exactly F
    (Section 4 of the assignment).
    """
    num_subcarriers = time_blocks.shape[0]
    return np.fft.fft(time_blocks, axis=0) / np.sqrt(num_subcarriers)


# ---------------------------------------------------------------------------
# (b) Equalization and (c) demapping
# ---------------------------------------------------------------------------
def equalize(received_grid, channel_response):
    """One-tap equalizer on every subcarrier: d^_i[k] = y~_i[k] / H[k]."""
    return received_grid / channel_response[:, np.newaxis]


def demap_resource_grid(grid):
    """Inverse of the grid mapping: s^[N i + k] = d^_i[k] (column after column)."""
    return grid.T.reshape(-1)


def ofdm_receiver(received, channel_response, num_subcarriers, cp_length,
                  num_ofdm_symbols):
    """The complete receiver of Fig. 3.

    Returns the estimate s^ and, for the plots, the grids y~ (after the DFT)
    and d^ (after the equalizer), both N x 14.
    """
    blocks = split_into_blocks(received, num_subcarriers, cp_length, num_ofdm_symbols)
    time_blocks = remove_cyclic_prefix(blocks, cp_length)
    received_grid = ofdm_demodulate(time_blocks)
    equalized_grid = equalize(received_grid, channel_response)
    return demap_resource_grid(equalized_grid), received_grid, equalized_grid


# ---------------------------------------------------------------------------
# Detection (not part of Fig. 3; used to count symbol errors)
# ---------------------------------------------------------------------------
def detect_symbols(estimates, constellation):
    """Minimum-distance (ML) decision: the closest constellation point.

    Same detector as in Assignment 2; distances to all points are computed at
    once as a (number of estimates) x (constellation size) array.
    """
    distances = np.abs(estimates[:, np.newaxis] - constellation[np.newaxis, :]) ** 2
    return constellation[np.argmin(distances, axis=1)]


# ---------------------------------------------------------------------------
# Check: noise-free loopback through transmitter, channel and receiver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    for folder in ["", "Task-2", "Task-3"]:
        sys.path.append(str(PROJECT_ROOT / folder))
    from parameters import (CHANNEL_PROFILES, CHANNEL_SEEDS, DATA_SEED, NUM_DATA_SYMBOLS,
                            NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS)
    from transmitter import generate_data, ofdm_transmitter, unitary_dft_matrix
    from channel import channel_taps, frequency_response, multipath_channel

    data = generate_data(NUM_DATA_SYMBOLS, np.random.default_rng(DATA_SEED))
    no_noise_snr = np.inf                    # noise variance 1/SNR = 0

    lines = ["Task 4: receiver check without noise (y~ should equal H d~ exactly)", ""]
    for scenario, profile in CHANNEL_PROFILES.items():
        taps = channel_taps(profile["tap_delays"], profile["tap_powers"],
                            np.random.default_rng(CHANNEL_SEEDS[scenario]))
        response = frequency_response(taps, NUM_SUBCARRIERS)
        designed_cp = len(taps) - 1

        # With the designed CP and with a CP one sample too short
        for cp_length in [designed_cp, designed_cp - 1]:
            x, grid = ofdm_transmitter(data, NUM_SUBCARRIERS, cp_length)
            y = multipath_channel(x, taps, no_noise_snr, np.random.default_rng(0))
            estimate, received_grid, _ = ofdm_receiver(y, response, NUM_SUBCARRIERS,
                                                       cp_length, NUM_OFDM_SYMBOLS)
            channel_error = np.max(np.abs(received_grid - response[:, np.newaxis] * grid))
            data_error = np.max(np.abs(estimate - data))
            lines.append(f"Scenario {scenario}, Ncp = {cp_length:2d}: "
                         f"max |y~ - H d~| = {channel_error:.2e},  "
                         f"max |s^ - s| = {data_error:.2e}")
        lines.append("")

    # The fft-based DFT must equal the matrix product F y_c
    rng = np.random.default_rng(1)
    test_block = (rng.standard_normal((NUM_SUBCARRIERS, 3))
                  + 1j * rng.standard_normal((NUM_SUBCARRIERS, 3)))
    dft_error = np.max(np.abs(ofdm_demodulate(test_block)
                              - unitary_dft_matrix(NUM_SUBCARRIERS) @ test_block))
    lines.append(f"max |fft / sqrt(N) - F y_c| = {dft_error:.2e}")

    print("\n".join(lines))
    results_folder = Path(__file__).resolve().parent / "results"
    results_folder.mkdir(exist_ok=True)
    (results_folder / "receiver_check.txt").write_text("\n".join(lines) + "\n")
