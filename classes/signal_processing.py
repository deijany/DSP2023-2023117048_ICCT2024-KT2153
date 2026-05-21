import numpy as np


class SNR:
    def __init__(self, X, X_est, remove_DC=True, grouping_method='mean'):
        self.remove_DC = remove_DC
        self.X = X
        self.X_est = X_est
        self.grouping_method = grouping_method
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
            X_est = np.expand_dims(X_est, axis=0)
            self.result = self._simple_signal(X, X_est)
        elif X.ndim == 2 and len(X) == 1:
            self.result = self._simple_signal(X, X_est)
        elif X.ndim == 2 and len(X) >= 1:
            self.result = self._multiple_signal(X, X_est)
        else:
            raise ValueError('Input dimensions not supported')

    def _simple_signal(self, X, X_est):
        if self.remove_DC:
            X_est = X_est - np.mean(X_est)
        num = np.sum(np.abs(X)**2)
        den = np.sum(np.abs(X_est - X)**2)
        SNR = 10 * np.log10(num / den)
        return SNR

    def _multiple_signal(self, X, X_est):
        SNR_est = []
        for position in range(0, len(X), 1):
            x = np.expand_dims(X[position], axis=0)
            x_est = np.expand_dims(X_est[position], axis=0)
            SNR_est.append(self._simple_signal(x, x_est))
        if self.grouping_method == 'min':
            SNR = np.min(SNR_est)
        elif self.grouping_method == 'max':
            SNR = np.max(SNR_est)
        elif self.grouping_method == 'mean':
            SNR = np.mean(SNR_est)
        else:
            SNR = SNR_est
        return SNR


class Statistics:
    @staticmethod
    def variance(vector):
        """Calculate the variance of array"""
        mean = sum(vector) / len(vector)
        return sum((x - mean) ** 2 for x in vector) / len(vector)

    @staticmethod
    def std_deviation(vector):
        """Calculate the standard deviation of an array."""
        return Statistics.variance(vector) ** 0.5
