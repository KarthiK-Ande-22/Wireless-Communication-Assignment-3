"""System parameters of the OFDM link (Table 2) and the two channels (Table 1).

Every script imports its numbers from this file, so each value is written
only once.
"""

# ---------------------------------------------------------------------------
# OFDM numerology (Table 2)
# ---------------------------------------------------------------------------
BANDWIDTH = 15.36e6                          # W in Hz, also the sampling rate
SAMPLE_PERIOD = 1 / BANDWIDTH                # about 65.1 ns
NUM_SUBCARRIERS = 512                        # N, the DFT size
SUBCARRIER_SPACING = BANDWIDTH / NUM_SUBCARRIERS   # 30 kHz
NUM_OFDM_SYMBOLS = 14                        # OFDM symbols in one slot
NUM_DATA_SYMBOLS = NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS   # 14 x 512 = 7168

SNR_DB = 30                                  # SNR per data subcarrier
SNR = 10 ** (SNR_DB / 10)                    # linear scale, 1000

# OFDM symbol used for the scatter plots of Task 5(c) (middle of the slot)
PLOT_SYMBOL = 7

# ---------------------------------------------------------------------------
# Multipath channels (Table 1): non-zero tap indices and their powers
# ---------------------------------------------------------------------------
CHANNEL_PROFILES = {
    1: {
        "tap_delays": [0, 1, 2, 4],
        "tap_powers": [0.539, 0.271, 0.136, 0.054],
    },
    2: {
        "tap_delays": [0, 2, 5, 10, 20, 35, 55, 75],
        "tap_powers": [0.277, 0.219, 0.174, 0.138, 0.110, 0.069, 0.009, 0.004],
    },
}

# CP length used in Task 5(d) ("insufficient CP")
SHORT_CP_LENGTH = 4

# ---------------------------------------------------------------------------
# Random seeds. Each random block has its own generator, so the data and the
# channel stay exactly the same when only the CP length is changed. Each
# scenario has its own channel seed, so the two channels are independent.
# ---------------------------------------------------------------------------
DATA_SEED = 2026                     # 4-QAM data symbols
CHANNEL_SEEDS = {1: 2027, 2: 2028}   # channel taps h of Scenario 1 and 2
NOISE_SEED = 2029                    # receiver noise w
