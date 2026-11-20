"""Task 4: OFDM receiver of Fig. 3 (the receiver knows h). Run this file for a noise-free check."""
import sys
from pathlib import Path

import numpy as np


# (a) OFDM demodulation

def split_into_blocks(received, num_subcarriers, cp_length, num_ofdm_symbols):
    """Cut the first 14 (N + Ncp) samples of y into blocks y_i, one per column; the tail is not used."""
    block_length = num_subcarriers + cp_length
    used_samples = received[:num_ofdm_symbols * block_length]
    return used_samples.reshape(num_ofdm_symbols, block_length).T


def remove_cyclic_prefix(blocks, cp_length):
    """Drop the first Ncp samples of every block, which leaves y_c,i with N samples."""
    return blocks[cp_length:, :]


def ofdm_demodulate(time_blocks):
    """Return y~_i = F y_c,i for every column; numpy's fft has no 1/sqrt(N), so F = fft / sqrt(N)."""
    num_subcarriers = time_blocks.shape[0]
    return np.fft.fft(time_blocks, axis=0) / np.sqrt(num_subcarriers)


# (b) Equalization

def equalize(received_grid, channel_response):
    """One-tap equalizer: d^_i[k] = y~_i[k] / H[k], the same H[k] for all 14 symbols."""
    return received_grid / channel_response[:, np.newaxis]


# (c) Demapping

def demap_resource_grid(grid):
    """Undo the grid mapping: s^[N i + k] = d^_i[k], column after column."""
    return grid.T.reshape(-1)


def ofdm_receiver(received, channel_response, num_subcarriers, cp_length, num_ofdm_symbols):
    """Run the whole receiver of Fig. 3; return s^ and the grids y~ and d^ (for the plots)."""
    blocks = split_into_blocks(received, num_subcarriers, cp_length, num_ofdm_symbols)
    time_blocks = remove_cyclic_prefix(blocks, cp_length)
    received_grid = ofdm_demodulate(time_blocks)
    equalized_grid = equalize(received_grid, channel_response)
    return demap_resource_grid(equalized_grid), received_grid, equalized_grid


# Extra, not part of Fig. 3: decision, used in Task 5 to count symbol errors

def detect_symbols(estimates, constellation):
    """Return the closest 4-QAM point to every estimate (minimum-distance detector of Assignment 2)."""
    distances = np.abs(estimates[:, np.newaxis] - constellation[np.newaxis, :]) ** 2
    return constellation[np.argmin(distances, axis=1)]


# Check: noise-free loopback through transmitter, channel and receiver

def check_loopback(data, taps, cp_length, num_subcarriers, num_ofdm_symbols):
    """Send one slot without noise; return max |y~ - H d~| and max |s^ - s|."""
    response = frequency_response(taps, num_subcarriers)
    x, grid = ofdm_transmitter(data, num_subcarriers, cp_length)
    y = multipath_channel(x, taps, np.inf, np.random.default_rng(0))     # 1/SNR = 0: no noise
    estimate, received_grid, _ = ofdm_receiver(y, response, num_subcarriers, cp_length,
                                               num_ofdm_symbols)
    channel_error = np.max(np.abs(received_grid - response[:, np.newaxis] * grid))
    data_error = np.max(np.abs(estimate - data))
    return channel_error, data_error


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    for folder in ["", "Task-2", "Task-3"]:                  # parameters, transmitter, channel
        sys.path.append(str(project_root / folder))
    from parameters import (CHANNEL_PROFILES, CHANNEL_SEEDS, DATA_SEED, NUM_DATA_SYMBOLS,
                            NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS)
    from transmitter import generate_data, ofdm_transmitter, unitary_dft_matrix
    from channel import channel_taps, frequency_response, multipath_channel

    data = generate_data(NUM_DATA_SYMBOLS, np.random.default_rng(DATA_SEED))
    lines = ["Task 4: receiver check without noise (y~ should equal H d~ exactly)", ""]

    for scenario, profile in CHANNEL_PROFILES.items():
        taps = channel_taps(profile["tap_delays"], profile["tap_powers"],
                            np.random.default_rng(CHANNEL_SEEDS[scenario]))
        designed_cp = len(taps) - 1
        for cp_length in [designed_cp, designed_cp - 1]:     # designed CP and one sample shorter
            channel_error, data_error = check_loopback(data, taps, cp_length, NUM_SUBCARRIERS,
                                                       NUM_OFDM_SYMBOLS)
            lines.append(f"Scenario {scenario}, Ncp = {cp_length:2d}: "
                         f"max |y~ - H d~| = {channel_error:.2e},  max |s^ - s| = {data_error:.2e}")
        lines.append("")

    # fft / sqrt(N) must equal the matrix product F y_c
    rng = np.random.default_rng(1)
    test_blocks = (rng.standard_normal((NUM_SUBCARRIERS, 3))
                   + 1j * rng.standard_normal((NUM_SUBCARRIERS, 3)))
    dft_error = np.max(np.abs(ofdm_demodulate(test_blocks)
                              - unitary_dft_matrix(NUM_SUBCARRIERS) @ test_blocks))
    lines.append(f"max |fft / sqrt(N) - F y_c| = {dft_error:.2e}")

    print("\n".join(lines))
    results_folder = Path(__file__).resolve().parents[1] / "results" / "Task-4"
    results_folder.mkdir(parents=True, exist_ok=True)
    (results_folder / "receiver_check.txt").write_text("\n".join(lines) + "\n")
