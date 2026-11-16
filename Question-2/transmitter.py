"""Task 2: OFDM transmitter (Fig. 2 of the assignment).

    s --> resource grid mapping --> IDFT F^H --> add CP --> concatenate --> x

Each block is a separate function. Running this file checks every block:

    python3 Question-2/transmitter.py
"""
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# (a) Data generation
# ---------------------------------------------------------------------------
def qam4_constellation():
    """4-QAM points (+-1 +- j) / sqrt(2), scaled to unit average energy.

    Same constellation as in Assignment 2.
    """
    points = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def generate_data(num_symbols, rng):
    """Random, equally likely 4-QAM symbols s[0], ..., s[num_symbols - 1]."""
    constellation = qam4_constellation()
    indices = rng.integers(0, len(constellation), size=num_symbols)
    return constellation[indices]


# ---------------------------------------------------------------------------
# (b) Resource grid mapping
# ---------------------------------------------------------------------------
def map_to_resource_grid(data, num_subcarriers):
    """Arrange s on the N x 14 grid:  grid[k, i] = d~_i[k] = s[N i + k].

    Column i holds the N values of OFDM symbol i: the first N data symbols go
    to symbol 0, the next N to symbol 1, and so on.
    """
    num_ofdm_symbols = len(data) // num_subcarriers
    return data.reshape(num_ofdm_symbols, num_subcarriers).T


# ---------------------------------------------------------------------------
# (c) OFDM modulation: IDFT, cyclic prefix, concatenation
# ---------------------------------------------------------------------------
def unitary_dft_matrix(size):
    """The N x N unitary DFT matrix, [F]_{k,m} = exp(-j 2 pi k m / N) / sqrt(N)."""
    index = np.arange(size)
    return np.exp(-2j * np.pi * np.outer(index, index) / size) / np.sqrt(size)


def ofdm_modulate(grid):
    """d_i = F^H d~_i for every OFDM symbol (every column of the grid).

    numpy's ifft includes a factor 1/N, so sqrt(N) * ifft is exactly the
    unitary IDFT F^H (Section 4 of the assignment).
    """
    num_subcarriers = grid.shape[0]
    return np.sqrt(num_subcarriers) * np.fft.ifft(grid, axis=0)


def add_cyclic_prefix(time_symbols, cp_length):
    """Put a copy of the last Ncp samples of every column in front of it.

    x_i = (d_i[N - Ncp], ..., d_i[N - 1], d_i[0], ..., d_i[N - 1]),
    so every column grows from N to N + Ncp samples.
    """
    num_subcarriers = time_symbols.shape[0]
    cyclic_prefix = time_symbols[num_subcarriers - cp_length:, :]
    return np.vstack([cyclic_prefix, time_symbols])


def concatenate_symbols(blocks):
    """Send the OFDM symbols one after the other: x = (x_0, x_1, ..., x_13).

    blocks has one OFDM symbol (with CP) per column; the result has
    M = 14 (N + Ncp) samples.
    """
    return blocks.T.reshape(-1)


def ofdm_transmitter(data, num_subcarriers, cp_length):
    """The complete transmitter of Fig. 2. Returns x and the resource grid."""
    grid = map_to_resource_grid(data, num_subcarriers)
    time_symbols = ofdm_modulate(grid)
    blocks = add_cyclic_prefix(time_symbols, cp_length)
    return concatenate_symbols(blocks), grid


# ---------------------------------------------------------------------------
# Check of every block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT))       # for parameters.py
    from parameters import DATA_SEED, NUM_DATA_SYMBOLS, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS

    cp_length = 75
    rng = np.random.default_rng(DATA_SEED)
    data = generate_data(NUM_DATA_SYMBOLS, rng)
    x, grid = ofdm_transmitter(data, NUM_SUBCARRIERS, cp_length)

    # (b) mapping: every entry grid[k, i] must hold s[N i + k]
    mapping_ok = all(grid[k, i] == data[NUM_SUBCARRIERS * i + k]
                     for i in range(NUM_OFDM_SYMBOLS) for k in range(NUM_SUBCARRIERS))

    # (c) the fft-based IDFT must equal the matrix product F^H d~
    F = unitary_dft_matrix(NUM_SUBCARRIERS)
    time_symbols = ofdm_modulate(grid)
    idft_error = np.max(np.abs(time_symbols - F.conj().T @ grid))
    unitary_error = np.max(np.abs(F @ F.conj().T - np.eye(NUM_SUBCARRIERS)))

    # (c) the CP of every OFDM symbol must equal its last Ncp samples
    block_length = NUM_SUBCARRIERS + cp_length
    blocks = x.reshape(NUM_OFDM_SYMBOLS, block_length)
    cp_error = np.max(np.abs(blocks[:, :cp_length] - blocks[:, -cp_length:]))

    # Energy: the unitary IDFT keeps the energy, so E|x[m]|^2 = E|s[n]|^2 = 1
    lines = [
        f"Task 2: transmitter check (CP length {cp_length}, data seed {DATA_SEED})",
        f"  number of data symbols            : {len(data)}",
        f"  average energy of s, E|s|^2       : {np.mean(np.abs(data) ** 2):.6f}",
        f"  resource grid shape (N x symbols) : {grid.shape}",
        f"  grid[k, i] == s[N i + k] for all  : {mapping_ok}",
        f"  max |F F^H - I|                   : {unitary_error:.2e}",
        f"  max |sqrt(N) ifft - F^H d~|       : {idft_error:.2e}",
        f"  length of x, M = 14 (N + Ncp)     : {len(x)}",
        f"  max |CP - last Ncp samples|       : {cp_error:.2e}",
        f"  average power of x, mean |x[m]|^2 : {np.mean(np.abs(x) ** 2):.4f}",
    ]
    print("\n".join(lines))
    results_folder = Path(__file__).resolve().parent / "results"
    results_folder.mkdir(exist_ok=True)
    (results_folder / "transmitter_check.txt").write_text("\n".join(lines) + "\n")
