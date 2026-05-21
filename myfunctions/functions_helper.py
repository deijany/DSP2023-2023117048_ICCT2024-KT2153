import matplotlib.pyplot as plt
import pickle
from scipy import signal
import numpy as np

import shutil
plt.rcParams['text.usetex'] = shutil.which('latex') is not None
# import matplotlib
# matplotlib.use('TkAgg')  # 'Qt5Agg' or 'TkAgg', 'GTK3Agg', etc.

def store_dictionary(base_filename, dictionary):
    """
    Save dictionary base_name to a pickle file.
    Parameters:
        base_filename (str): The base path and name of the files before extension.
        dictionary (dict): The dictionary to be pickled and saved.
    Returns:
        None
    """
    pickle_filename = base_filename + ".pickle"
    # Save to pickle file
    with open(pickle_filename, "wb") as f:
        pickle.dump(dictionary, f)


class SpectrumAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def get_window(window_type='Blackmanharris', length=100):
        """
        Get a window function of a specified type and length.

        Parameters:
        - window_type: str, the type of window to generate. Options are 'Blackmanharris', 'Hamming', and 'Rectangular'.
        - length: int, the length of the window.

        Returns:
        - window: numpy array, the window function.
        """
        if window_type == 'Blackmanharris':
            window = signal.windows.blackmanharris(length, sym=True)
        elif window_type == 'Hamming':
            window = signal.windows.hamming(length, sym=True)
        elif window_type == 'Rectangular':
            window = np.ones(length)
        else:
            # If an incorrect window is specified, use Blackmanharris and raise a warning
            window = signal.windows.blackmanharris(length, sym=True)
            print('Warning: Incorrect window type introduced. Continuing with Blackmanharris window.')
        return window

    @staticmethod
    def plot_frequency_domain(signal_matrix, title, window_type, save_path, x_rfft_max=None, save_fig=False):
        '''
        :param XM: Matrix with signals of size (n_signal, n_samples)
        :param title:
        :param window_type: Blackmanharris or Rectangular
        :param save_path: directory to save the plots
        :param x_rfft_max: For relative normalization
        :param save_fig:
        :return:
        '''
        fs = 1
        XM = np.copy(signal_matrix)

        # Set the figure width to 3.5 inches and height to maintain a 4:3 aspect ratio
        fig_width = 3.7  # inches
        fig_height = fig_width #* (3.0 / 4.0)  # 4:3 aspect ratio

        fig, axs = plt.subplots(nrows=len(XM), ncols=1, figsize=(fig_width, fig_height))
        for ii, X in enumerate(XM):
            axs[ii].set_ylabel(r'Amplitude spectrum (dB)', fontsize=8)
            axs[ii].set_xlabel(r'Normalized frequency', fontsize=8)
            axs[ii].set_title(r'\textbf{{{}}}'.format(title[ii]), fontsize=8)
            N = 16
            SNR_lim = 6.02 * N + 1.76
            axs[ii].set_ylim((-SNR_lim - 15, 0.5))
            axs[ii].margins(x=0)
            plt.subplots_adjust(hspace=0.5)
            for k in range(0, len(X), 1):
                font_dict = {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}
                # axs[ii].set_title(title[ii], fontdict=font_dict)
                if X.ndim == 1:
                    X = np.expand_dims(X, axis=0)

                sequence = X[k]  # Assuming you want to process each element of X
                window = SpectrumAnalyzer.get_window(window_type, len(sequence))
                vector_value = sequence * window
                rfft = np.abs(np.fft.rfft(vector_value))
                rfft_max = max(rfft)
                if x_rfft_max is None:
                    x_rfft_max = rfft_max
                p = 20 * np.log10(rfft / rfft_max)
                f = np.linspace(0, fs / 2, len(p))
                # Frequency Domain
                # axs[ii].set_title(title[ii], fontdict=font_dict)
                axs[ii].plot(f / max(f), p, linewidth=0.35)
                axs[ii].set_ylim([-75, 0])



        if save_fig:
            disk_path = str(save_path)
        plt.savefig(disk_path + '.pdf', bbox_inches='tight', pad_inches=0.05, transparent=True)
        return
def print_snr_summary(X_train, V_train, X_hat_train, SNR_array_X_train, SNR_array_V_train, SNR_array_X_hat_train,
                      X_test, V_test, X_hat_test, SNR_array_X_test, SNR_array_V_test, SNR_array_X_hat_test,
                      X, V, ak, L2, bk):
    print('-'*60)
    print('-'*25+'Prediction'+'-'*25)
    print('Train X:', '          (', len(X_train), ' signals)', ' min:', np.round(np.min(SNR_array_X_train), 2), ' mean:', np.round(np.mean(SNR_array_X_train), 2), ' max:', np.round(np.max(SNR_array_X_train), 2))
    print('Train V:', '          (', len(V_train), ' signals)', ' min:', np.round(np.min(SNR_array_V_train), 2), ' mean:', np.round(np.mean(SNR_array_V_train), 2), ' max:', np.round(np.max(SNR_array_V_train), 2))
    print('Train MI:', '         (', len(X_hat_train), ' signals)', ' min:', np.round(np.min(SNR_array_X_hat_train), 2), ' mean:', np.round(np.mean(SNR_array_X_hat_train), 2), ' max:', np.round(np.max(SNR_array_X_hat_train), 2))
    print('-'*35)
    print('Test X:', '           (', len(X_test), ' signals)', ' min:', np.round(np.min(SNR_array_X_test), 2), ' mean:', np.round(np.mean(SNR_array_X_test), 2), ' max:', np.round(np.max(SNR_array_X_test), 2))
    print('Test V:', '           (', len(V_test), ' signals)', ' min:', np.round(np.min(SNR_array_V_test), 2), ' mean:', np.round(np.mean(SNR_array_V_test), 2), ' max:', np.round(np.max(SNR_array_V_test), 2))
    print('Test MI:', '           (', len(X_hat_test), ' signals)', ' min:', np.round(np.min(SNR_array_X_hat_test), 2), ' mean:', np.round(np.mean(SNR_array_X_hat_test), 2), ' max:', np.round(np.max(SNR_array_X_hat_test), 2))
    print('-'*60)
    print(f"X: [{np.min(X):.2f}, {np.max(X):.2f}]  V: [{np.min(V):.2f}, {np.max(V):.2f}]  X_hat_train: [{np.min(X_hat_train):.2f}, {np.max(X_hat_train):.2f}]  X_hat_test: [{np.min(X_hat_test):.2f}, {np.max(X_hat_test):.2f}]")
    print('ak: max-->', np.max(abs(ak)), '  L2:', L2, '  bk_max:', np.max(abs(bk)))


