"""Task 2: OFDM transmitter of Fig. 2. Run this file to check every block."""
import sys
from pathlib import Path

import numpy as np


# (a) Data generation

def qam4_constellation():
    """Return the 4-QAM points (+-1 +- j) / sqrt(2), with unit average energy."""
    return np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)


def generate_data(num_symbols, rng):
    """Return num_symbols random, equally likely 4-QAM symbols s[n]."""
    constellation = qam4_constellation()
    indices = rng.integers(0, len(constellation), size=num_symbols)
    return constellation[indices]


# (b) Resource grid mapping

def map_to_resource_grid(data, num_subcarriers):
    """Return the N x 14 grid with grid[k, i] = d~_i[k] = s[N i + k]."""
    num_ofdm_symbols = len(data) // num_subcarriers
    return data.reshape(num_ofdm_symbols, num_subcarriers).T


# (c) OFDM modulation

def unitary_dft_matrix(size):
    """Return the N x N unitary DFT matrix, [F]_{k,m} = exp(-j 2 pi k m / N) / sqrt(N)."""
    index = np.arange(size)
    return np.exp(-2j * np.pi * np.outer(index, index) / size) / np.sqrt(size)


def ofdm_modulate(grid):
    """Return d_i = F^H d~_i for every column; numpy's ifft has 1/N, so F^H = sqrt(N) ifft."""
    num_subcarriers = grid.shape[0]
    return np.sqrt(num_subcarriers) * np.fft.ifft(grid, axis=0)


def add_cyclic_prefix(time_symbols, cp_length):
    """Put a copy of the last Ncp samples of every column in front of it."""
    num_subcarriers = time_symbols.shape[0]
    cyclic_prefix = time_symbols[num_subcarriers - cp_length:, :]
    return np.vstack([cyclic_prefix, time_symbols])


def concatenate_symbols(blocks):
    """Send the columns one after the other: x = (x_0, x_1, ..., x_13)."""
    return blocks.T.reshape(-1)


def ofdm_transmitter(data, num_subcarriers, cp_length):
    """Run the whole transmitter of Fig. 2 and return x and the resource grid."""
    grid = map_to_resource_grid(data, num_subcarriers)
    time_symbols = ofdm_modulate(grid)
    blocks = add_cyclic_prefix(time_symbols, cp_length)
    return concatenate_symbols(blocks), grid


# Check of every block

def check_transmitter(data, num_subcarriers, cp_length):
    """Check the mapping, IDFT, CP and concatenation for one CP length."""
    x, grid = ofdm_transmitter(data, num_subcarriers, cp_length)
    num_ofdm_symbols = grid.shape[1]
    block_length = num_subcarriers + cp_length

    # (b) every grid[k, i] must hold s[N i + k]
    mapping_ok = all(grid[k, i] == data[num_subcarriers * i + k]
                     for i in range(num_ofdm_symbols) for k in range(num_subcarriers))

    # (c) sqrt(N) ifft must equal the matrix product F^H d~
    dft_matrix = unitary_dft_matrix(num_subcarriers)
    time_symbols = ofdm_modulate(grid)
    idft_error = np.max(np.abs(time_symbols - dft_matrix.conj().T @ grid))

    # (c) unitary F keeps the energy: ||d_i||^2 = ||d~_i||^2 = N
    symbol_energy = np.sum(np.abs(time_symbols) ** 2, axis=0)

    # (c) block i of x must be (d_i[N - Ncp], ..., d_i[N - 1], d_i[0], ..., d_i[N - 1])
    blocks = x.reshape(num_ofdm_symbols, block_length)
    cp_error = np.max(np.abs(blocks[:, :cp_length] - blocks[:, num_subcarriers:]))
    order_error = np.max(np.abs(blocks[:, cp_length:] - time_symbols.T))

    return [
        f"CP length Ncp = {cp_length}",
        f"  grid shape (N x symbols)          : {grid.shape}",
        f"  grid[k, i] == s[N i + k] for all  : {mapping_ok}",
        f"  max |sqrt(N) ifft - F^H d~|       : {idft_error:.2e}",
        f"  energy of d_i, min and max (N)    : {symbol_energy.min():.6f}, {symbol_energy.max():.6f}",
        f"  max |CP - last Ncp samples|       : {cp_error:.2e}",
        f"  max |x block i without CP - d_i|  : {order_error:.2e}",
        f"  length of x, M = 14 (N + Ncp)     : {len(x)} (expected {num_ofdm_symbols * block_length})",
        f"  average power of x, mean |x[m]|^2 : {np.mean(np.abs(x) ** 2):.4f}",
    ]


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))   # for parameters.py
    from parameters import DATA_SEED, NUM_DATA_SYMBOLS, NUM_SUBCARRIERS

    data = generate_data(NUM_DATA_SYMBOLS, np.random.default_rng(DATA_SEED))
    dft_matrix = unitary_dft_matrix(NUM_SUBCARRIERS)
    unitary_error = np.max(np.abs(dft_matrix @ dft_matrix.conj().T - np.eye(NUM_SUBCARRIERS)))

    lines = [
        f"Task 2: transmitter check (data seed {DATA_SEED})",
        f"  number of data symbols, 14 x 512  : {len(data)}",
        f"  average energy of s, E|s|^2       : {np.mean(np.abs(data) ** 2):.6f}",
        f"  max |F F^H - I|                   : {unitary_error:.2e}",
    ]
    for cp_length in (4, 75):        # designed CP of Scenario 1 and Scenario 2
        lines += check_transmitter(data, NUM_SUBCARRIERS, cp_length)

    print("\n".join(lines))
    results_folder = Path(__file__).resolve().parents[1] / "results" / "Task-2"
    results_folder.mkdir(parents=True, exist_ok=True)
    (results_folder / "transmitter_check.txt").write_text("\n".join(lines) + "\n")
