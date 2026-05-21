from classes.file_manipulation import PathManager, DataSetLoader, DataSetWriter
from classes.generator_classes_v4 import PolynomialSignalGenerator
from myfunctions.functions_helper import SpectrumAnalyzer, store_dictionary, print_snr_summary
from classes.linearizers_classes import MatrixInversionLinearizer, ActivationFunctions
from classes.signal_processing import Statistics, SNR
import numpy as np

np.random.seed(1)

# --- Dataset ---
dataset_features = {
    'amount_signals': 100,
    'amount_sinusoids_per_signal': 31,
    'equidistant_sinusoids_per_signal': True,
    'polynomial_order': 9,
    'number_samples_signal': int(np.power(2, 13)),
    'amount_bits': 8,
    'frequency_dependent_case': False,
    'initial_frequency': 47e6,
    'last_frequency': 1576e6,
    'fs': 3400e6,
    'bandwidth_percent_on_dataset': 40,
    'initial_frequency_gap': 20,
    'generator_method': 'polynomial_model'
}
dataset_version = 1
storage_dataset = False  # set True to regenerate dataset from scratch

# --- Linearizer ---
activation_function = ActivationFunctions.ReLU  # change here: ABS, ReLU, Q1b, linear, polynomial
activation_name = activation_function.__name__

branch_number = 7
output_resolution = 12
internal_resolution = 1000
L2 = 1e-10
number_simulations = 10
plotting = True
store_training = True

# --- Signal Generation ---
signals = PolynomialSignalGenerator(dataset_features, seed_value=7)
X, _, _ = signals.generate_multisine()
X = X / np.max(abs(X), axis=1)[:, np.newaxis]
X = X * 0.94
V, nonlinear_coefficients = signals.generate_distorted_signals(X)

folder_prefix = 'v' + str(dataset_version)
current_path, _ = PathManager().check_path_by_host()
dataset_path = ('v' + str(dataset_version) + '_f' + 'freq_dep' + '0_ord_'
                + str(dataset_features['amount_bits']) + 'bits_'
                + str(dataset_features['polynomial_order'] + 1) + 'terms')
train_path, test_path = PathManager().make_path(root_path=current_path, local_path='datasets',
                                                 current_path=dataset_path, state='data')
simulation_path = PathManager().make_path(root_path=current_path, local_path='trained_model',
                                           current_path=folder_prefix, state='simulations')
if storage_dataset:
    train_percent = 0.1
    dataset = DataSetWriter(dataset_features, train_path, test_path, train_pct=train_percent)
    dataset.write_dataset(V, X, nonlinear_coefficients)
    print("new dataset was storated.")
else:
    print("using storated dataset")

# --- Load Data ---
loader1 = DataSetLoader(path=train_path)
data1 = loader1.load_dataset()
X_train = data1['pure_signal']
V_train = data1['distorted_signal']

loader2 = DataSetLoader(path=test_path)
data2 = loader2.load_dataset()
X_test = data2['pure_signal']
V_test = data2['distorted_signal']

# --- SNR Baselines ---
X_train_quant = PolynomialSignalGenerator.quantize(X_train, dataset_features['amount_bits'])
X_test_quant = PolynomialSignalGenerator.quantize(X_test, dataset_features['amount_bits'])

SNR_array_X_test = SNR(X_test, X_test_quant, remove_DC=True, grouping_method=None).result
SNR_array_V_test = SNR(X_test, V_test, remove_DC=True, grouping_method=None).result
SNR_array_X_train = SNR(X_train, X_train_quant, remove_DC=True, grouping_method=None).result
SNR_array_V_train = SNR(X_train, V_train, remove_DC=True, grouping_method=None).result


# --- Train ---
linearizer = MatrixInversionLinearizer(X_train, V_train, branch_number, activation_function,
                                        number_simulations, L2,
                                        output_quantizer_resolution=output_resolution,
                                        internal_quantizer_resolution=internal_resolution)
ak, bk, _, _ = linearizer.train()

# --- Predict ---
# NOTE: if you sweep L2, re-initialize linearizer here with the best L2 before calling predictor,
# since the loop leaves linearizer pointing to the last L2 tried, not the best one.
X_hat_test_MI, _, _ = linearizer.predictor(X_test, V_test, ak, bk)
X_hat_train_MI, _, _ = linearizer.predictor(X_train, V_train, ak, bk)

SNR_array_X_hat_train_MI = SNR(X_train, X_hat_train_MI, remove_DC=True, grouping_method=None).result
SNR_array_X_hat_test_MI = SNR(X_test, X_hat_test_MI, remove_DC=True, grouping_method=None).result

# --- Results ---
dictionary = {
    'activation': activation_name,
    'branch_number': branch_number,
    'L2': L2,
    'ak': ak,
    'bk': bk,
    'SNR_V_test': np.mean(SNR_array_V_test),
    'SNR_X_test': np.mean(SNR_array_X_test),
    'SNR_X_hat_test_MI': np.mean(SNR_array_X_hat_test_MI),
    'SNR_V_train': np.mean(SNR_array_V_train),
    'SNR_X_train': np.mean(SNR_array_X_train),
    'SNR_X_hat_train_MI': np.mean(SNR_array_X_hat_train_MI),
    'variance_x_hat_SNR': Statistics.variance(SNR_array_X_hat_test_MI),
    'std_deviation_x_hat_SNR': Statistics.std_deviation(SNR_array_X_hat_test_MI),
}

print_snr_summary(X_train, V_train, X_hat_train_MI, SNR_array_X_train, SNR_array_V_train, SNR_array_X_hat_train_MI,
                  X_test, V_test, X_hat_test_MI, SNR_array_X_test, SNR_array_V_test, SNR_array_X_hat_test_MI,
                  X, V, ak, L2, bk)

if plotting:
    title = ['Distorted signal', 'Linearized signal']
    for pos in range(0, len(V_test), 4000):
        x = np.expand_dims(X_hat_test_MI[pos], axis=0)
        v = np.expand_dims(V_test[pos], axis=0)
        SpectrumAnalyzer().plot_frequency_domain([v, x], title, window_type='Blackmanharris',
                                                  save_path='./plots/' + activation_name + '_multisine_' + str(internal_resolution) + 'Ibits_' + str(output_resolution) + 'Obits',
                                                  save_fig=True)

if store_training:
    base_filename = (simulation_path + "/" + str(folder_prefix) + '_' + activation_name + '_ord_' + str(0)
                     + '_quant_' + str(internal_resolution) + 'Ibits_' + str(output_resolution) + 'Obits')
    store_dictionary(base_filename, dictionary)
else:
    print('Training was not saved')
