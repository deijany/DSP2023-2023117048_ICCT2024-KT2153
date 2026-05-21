import numpy as np


class PolynomialSignalGenerator:
    def __init__(self, dataset_features, seed_value):
        self.dataset_features = dataset_features
        self.amount_signals = dataset_features['amount_signals']
        self.amount_sinusoids_per_signal = dataset_features['amount_sinusoids_per_signal']
        self.polynomial_order = dataset_features['polynomial_order']
        self.number_samples_signal = dataset_features['number_samples_signal']
        self.amount_bits = dataset_features['amount_bits']
        self.frequency_dependent_case = dataset_features['frequency_dependent_case']
        np.random.seed(seed_value)

    @staticmethod
    def quantize(signals, amount_bits):
        # Assumes a full-scale signal x in the interval [-1,1)
        Q = 2**-(amount_bits-1)
        signals = signals * (1-1e-12)
        signals = signals - Q/2
        signals = np.round(signals * 2**(amount_bits-1)) / 2**(amount_bits-1)
        return signals + Q/2

    def _frequency_independent_coefficient(self, term_index):
        if term_index == 1:
            a_coef = [0, 1, 0]
        elif term_index <= 0:
            raise ValueError('term index should be integer')
        else:
            a_coef = (0.15 / term_index) * np.power(-1, term_index) * np.array([0, 1, 0])
        return a_coef

    def _frequency_dependent_coefficient(self, term_index):
        if term_index == 1:
            a_coef = [0, 1, 0]
        elif term_index <= 0:
            raise ValueError('term index should be integer')
        else:
            a_coef = np.array([1, -1, 1]) * 2 * np.array([1/100, 2/100, 0.01 * term_index / (1 + self.polynomial_order)])
        return a_coef

    def generate_multisine(self):
        """Generates a dataset of multisine signals."""
        amount_carriers = 31
        subcarrier_frequencies = np.arange(1, amount_carriers + 1)

        offsets = 0.17 * 2 * np.pi / 64
        freq_grid = 2 * np.pi * subcarrier_frequencies / (2 * (1 + amount_carriers)) + offsets
        fs = 2 * np.pi

        multisine_signals = np.zeros((self.amount_signals, self.number_samples_signal))
        frequency_array = np.zeros((self.amount_signals, len(subcarrier_frequencies)))
        amplitude_array = np.zeros((self.amount_signals, len(subcarrier_frequencies)))

        for i in range(self.amount_signals):
            for j, frequency in enumerate(freq_grid):
                amplitude = 1
                phase = np.random.choice([1, 3, 5, 7], replace=True) * np.pi / 4
                sine_wave = amplitude * np.sin(2 * np.pi * frequency * np.arange(self.number_samples_signal) / fs + phase)
                multisine_signals[i] += sine_wave
                frequency_array[i, j] = frequency
                amplitude_array[i, j] = amplitude
        return multisine_signals / self.amount_sinusoids_per_signal, frequency_array, amplitude_array

    def generate_distorted_signals(self, X):
        """Generates distorted signals V from pure signals X using a polynomial model:
        v = a1*x^1 + a2*x^2 + ... + an*x^n
        """
        pure_signals = X
        distorted_signals = np.zeros(np.shape(pure_signals))
        for i in range(self.amount_signals):
            v = 0
            if i == 0:
                nonlinear_coeff_vector = []
            for k_index in range(0, self.polynomial_order + 1):
                if self.frequency_dependent_case is False:
                    nonlinearity_coefficient = self._frequency_independent_coefficient(k_index + 1)
                else:
                    nonlinearity_coefficient = self._frequency_dependent_coefficient(k_index + 1)
                if i == 0:
                    nonlinear_coeff_vector.append(nonlinearity_coefficient)
                xk = np.power(pure_signals[i], k_index + 1)
                v += np.convolve(xk, nonlinearity_coefficient, 'same')
            distorted_signals[i] += v
        distorted_signals = self.quantize(distorted_signals, self.amount_bits)
        return distorted_signals, nonlinear_coeff_vector
