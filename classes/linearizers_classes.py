import numpy as np
# import random
import time
# import copy

class MatrixInversionLinearizer:
    def __init__(self, X_ref, V, branch_number, activation_function, number_simulations, L2_constant, output_quantizer_resolution=50, internal_quantizer_resolution=50):
        # Initialize the object with the following attributes:
        # - X_ref: reference signal(s)
        # - V: input signal(s)
        # - branch_number: number of branches in the network
        # - activation_function: activation function used in the network
        # - number_simulations: number of simulations to run to find the best solution
        # - L2_constant: constant used to regularize the matrix inversion
        self.X_ref = X_ref
        self.V = V
        self.branch_number = branch_number # these that going to activation function
        self.activation_function = activation_function
        self.number_simulations = number_simulations
        self.L2 = L2_constant
        self.output_quantizer_resolution = output_quantizer_resolution
        self.internal_quantizer_resolution = internal_quantizer_resolution

    def _bkgenerator(self, seed, mode):
        if mode == 'random':
            np.random.seed(seed+int(time.time()))
            bk = 2*(np.random.rand(self.branch_number)-0.5)
            
        elif mode == 'linesearch':
            a = -1
            b = 1
            step = 0.25*(b - a)/(self.number_simulations) # calculate the step size
            a = np.round(a + seed*step, 6) # Adjust a for current iteration
            b = np.round(b - seed*step, 6) # Adjust b for current iteration
            bk = np.linspace(a, b, self.branch_number, endpoint=True)
            # print('bk_value:', bk, 'branch_number:', self.branch_number)
        else:
            raise ValueError('Invalid bk generator method')
        bk = np.expand_dims(bk, 1)
        if self.activation_function == ActivationFunctions.polynomial:
            bk = abs(bk)*0
        return bk
        
    def _make_matrix(self, v, bk):
        # Helper function to create the matrix used in the network
        v_extra_bias = np.ones(np.shape(v))
        v_mod = np.concatenate([self.activation_function(v+bk), v, v_extra_bias], axis=0)
        return v_mod

    def _MRSE(self, y_true, y_pred, mode = 'mean'):
        # Helper function to calculate the mean root square error (MRSE) between the true signal and predicted signal
        diff = y_true - y_pred
        mse = np.mean(diff@diff.T, axis=1)  # calculate the mean of each row
        mrse = np.sqrt(mse)
        if mode == 'min':
            mrse=np.min(mrse)
        if mode == 'max':
            mrse=np.max(mrse)
        if mode == 'mean':
            mrse=np.mean(mrse)
        return mrse# return the minimum of the mean values
    
    def _ak_solver(self, current_bk):
        # Helper function to solve for the weights (ak) in the network
        # diff = np.expand_dims(np.sum(self.X_ref-self.V, axis = 0), axis = 0)
        for pos in range (0, len(self.V)):
            x_ref = np.expand_dims(self.X_ref[pos], axis=0)
            v = np.expand_dims(self.V[pos], axis=0)
            v_mod = self._make_matrix(v, current_bk)
            
            if pos == 0:
                A = v_mod@v_mod.T
                b = v_mod@(x_ref - v).T
            else:
                A += v_mod@v_mod.T
                b += v_mod@(x_ref - v).T
            A += len(self.V)*self.L2*np.identity(len(A))

        # print('rank:', np.linalg.matrix_rank(A))
        ak = np.linalg.inv(A)@b
        return ak
    
    def train(self):
        best_SNR = -np.inf
        if self.X_ref.ndim ==1:
            self.X_ref = np.expand_dims(self.X_ref[0], axis=0)
            self.V = np.expand_dims(self.V[0], axis=0)
        
        #Find best ak, bk for a signal
        best_ak = None
        best_error = np.inf
        best_SNR = -np.inf
        # best_bk = None
        for seed in range(self.number_simulations):
            if self.activation_function == ActivationFunctions.polynomial and seed > 0:
                continue
            # st_local_x_hat = None
            current_bk = self._bkgenerator(seed, mode='linesearch')
            current_ak = self._ak_solver(current_bk)
            current_x_hat, current_error , current_SNR_value = self.predictor(self.X_ref, self.V, current_ak, current_bk)
            if best_SNR < current_SNR_value:
                # best_x_hat = current_x_hat
                best_ak = current_ak
                best_bk = current_bk
                best_error = current_error
                best_SNR = current_SNR_value
            print('branch_number:', self.branch_number, '  Init_numb:',(seed+1),'/',self.number_simulations, ' best_error:', np.round(best_error,6),' best_SNR:', np.round(best_SNR,5) , ' current_SNR:', np.round(current_SNR_value,5))
        return best_ak, best_bk, best_error, best_SNR
    
    def predictor(self, X_ref, V, ak, bk):
        X_hat = np.zeros(np.shape(X_ref))
        for pos in range (0, len(X_ref)):
            # x_ref = np.expand_dims(X_ref[pos], axis=0)
            v = np.expand_dims(V[pos], axis=0)
            v_mod = self._make_matrix(v, bk)
            # X_hat[pos] = v + ActivationFunctions._quantize(ak.T@ActivationFunctions._quantize(v_mod, self.internal_quantizer_resolution), self.output_quantizer_resolution)

            X_hat[pos] = v +ActivationFunctions._quantize(ak, self.output_quantizer_resolution).T@v_mod
            
        
        error =  self._MRSE(X_ref, X_hat, mode='min')     
        SNR_value = SNR(X_ref,X_hat, remove_DC=False, grouping_method='min').result
        # print('-'*20)
        # print('SNR (minimum_value):', SNR_value)
        return X_hat, error, SNR_value
    

class ActivationFunctions:
    """Activation Functions for Neural Networks.

       Methods:
       - _quantize(signals, internal_quantizer_resolution_bits=50): Quantizes input signals.
       - linear(x): Linear activation function.
       - ReLU(x): Rectified Linear Unit (ReLU) activation function.
       - ABS(x): Absolute value activation function.
       - polynomial(x): Polynomial activation function.

       Usage:
       activation_instance = ActivationFunctions()
       quantized_signals = activation_instance._quantize(signals)
       linear_result = activation_instance.linear(x)
       relu_result = activation_instance.ReLU(x)
       abs_result = activation_instance.ABS(x)
       poly_result = activation_instance.polynomial(x)
       """

    @classmethod
    def _quantize(cls, signals, amount_bits=50):
        # Assumes a full-scale signal x in the interval [-1,1)
        Q = 2**-(amount_bits-1)
        signals = signals * (1-1e-12)
        signals = signals - Q/2
        signals = np.round(signals * 2**(amount_bits-1)) / 2**(amount_bits-1)
        return signals + Q/2

    @staticmethod
    def linear(x):
        """
        Linear activation function.
        """
        x = np.copy(x)
        return x

    @staticmethod
    def ReLU(x):
        """
        ReLU's activation function.
        """
        x = np.copy(x)
        return np.maximum(0, x)

    @staticmethod
    def ABS(x):
        x = x.copy()
        """
        Absolute value activation function.
        """
        return abs(x)
    
    @staticmethod
    def Q1b(x):
        """
        Sign activation function
        """
        x_copy = np.copy(x)
        x_copy[x_copy >= 0] = 1
        x_copy[x_copy < 0] = -1
        return x_copy.astype(int)
        

    @classmethod
    def polynomial(cls, x):
        """
        Polynomial activation function.
        """
        x_input = x[0].copy()
        # x_input = copy.deepcopy(x[0])

        for i in range(len(x)):
            if i == 0:
                # x[i] = cls._quantize(x[i]*x_input)
                x[i] = x[i]*x_input

            else:
                # x[i] = cls._quantize(x[i-1]*x_input)
                x[i] = x[i - 1] * x_input
        return x
        

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