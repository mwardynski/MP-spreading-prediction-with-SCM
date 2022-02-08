''' Spreading model based on Multivariate Autoregressive Model. 

Based on publication: 
A.Crimi et al. "Effective Brain Connectivity Through a Constrained Autoregressive Model" MICCAI 2016
'''

import os
from glob import glob
import logging
import random
import warnings
import concurrent.futures

from tqdm import tqdm 
import numpy as np
from scipy.stats.stats import pearsonr as pearson_corr_coef

from utils_vis import *
from utils import *

logging.basicConfig(level=logging.INFO)

class MARsimulation:
    def __init__(self, connect_matrix, t0_concentrations, t1_concentrations):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations. '''
        self.N_regions = 170                                                    # no. of brain areas from the atlas
        self.maxiter = int(1e6)                                                 # max no. of iterations for the gradient descent
        self.error_th = 0.01                                                    # acceptable error threshold for the reconstruction error
        self.gradient_th = 0.1                                                  # gradient difference threshold in stopping criteria in GD
        self.eta = 1e-7                                                         # learning rate of the gradient descent       
        self.cm = connect_matrix                                                # connectivity matrix 
        self.min_tract_num = 2                                                  # min no. of fibers to be kept (only when inverse_log==True)
        self.init_concentrations = t0_concentrations
        self.final_concentrations = t1_concentrations

    def run(self, norm_opt, inverse_log=True):
        ''' Run simulation. 
        
        Args:
            norm_opt (int): normalize option for connectivity matrix
            inverse_log (boolean): if True use normal values instead of logarithmic in connectivity matrix '''
        if inverse_log: 
            self.calc_exponent()
            self.filter_connections()
        self.transform_cm(norm_opt)
        self.generate_indicator_matrix()
        self.coef_matrix = self.run_gradient_descent() # get the model params
        pred_concentrations = self.coef_matrix @ self.init_concentrations # make predictions 
        return pred_concentrations
 
    def calc_exponent(self):
        ''' 
        Transform the connectivity matrix to get the no. of connections between regions
        instead of logarithm. 
        
        Inverse operation to log1p. '''
        self.cm = np.expm1(self.cm)
        
    def filter_connections(self):
        ''' Filter out all connections with less than n fiber reaching. '''
        self.cm = np.where(self.cm > self.min_tract_num, self.cm, 0).astype('float32')
        
    def transform_cm(self, norm_opt):
        '''
        Transform initial conenctivity matrix. 
        
        Args:
            0 means no normalization, = 1 means binarize, = 2 means divide by the maximum '''
        
        if norm_opt == 0:
            logging.info('No normalization of the initial matrix')
        elif norm_opt == 1:
            logging.info('Initial matrix binarized')
            self.cm = np.where(self.cm > 0, 1, 0).astype('float32')
        elif norm_opt == 2:
            logging.info('Initial matrix normalized according to its largest value')
            max_val = np.max(self.cm)
            self.cm /= max_val
        else:
            logging.info('No normalization of the initial matrix')
            
    def generate_indicator_matrix(self):
        ''' Construct a matrix with only zeros and ones to be used to 
        reinforce the zero connection (this is **B** in our paper).
        B has zero elements where no structural connectivity appears. '''
        self.B = np.where(self.cm==0, 0, 1).astype('float32')
        
    def run_gradient_descent(self, vis_error=False):
        iter_count = 0                                                          # counter of the current iteration 
        error_reconstruct = 1e10                                                # initial error of reconstruction gradients)
        if vis_error: error_buffer = []                                         # reconstruction error along iterations
        
        A = self.cm                                                             # the resulting effective matrix; initialized with connectivity matrix; [N_regions x N_regions]
        gradient = np.ones((self.N_regions, self.N_regions)) 
        
        while (error_reconstruct > self.error_th) and iter_count < self.maxiter:
            # calculate reconstruction error 
            error_reconstruct = 0.5 * np.linalg.norm(self.final_concentrations - (A * self.B) @ self.init_concentrations, ord=2)**2
            if vis_error: error_buffer.append(error_reconstruct)
            
            # gradient computation
            gradient = -(self.final_concentrations - (A * self.B) @ self.init_concentrations) @ (self.init_concentrations.T * self.B) 
            A -= self.eta * gradient 
            # reinforce where there was no connection at the beginning 
            A *= self.B
            norm = np.linalg.norm(gradient)
                    
            if norm < self.gradient_th:
                logging.info(f"Gradient norm: {norm}.\nTermination criterion met, quitting...")
                break    

            if iter_count % 100000 == 0:
                logging.info(f'Gradient norm at {iter_count}th iteration: {norm:.2f}')
            
            iter_count += 1
                              
        if vis_error: visualize_error(error_buffer)

        logging.info(f"Final reconstruction error: {error_reconstruct}")
        logging.info(f"Iterations: {iter_count}")
        
        return A
                   
def run_simulation(subject, plot=True, save_results=False):    
    ''' Run simulation for single patient. '''
    
    dataset_dir = '../../data/ADNI/derivatives/'
    output_dir = '../../results' 
      
    connectivity_matrix_path = os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'dwi', 'connect_matrix_rough.csv'))
    t0_concentration_path = sorted(glob(os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'pet', '*.csv'))))[0]
    t1_concentration_path = sorted(glob(os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-followup', 'pet', '*.csv'))))[0]
 
    subject_output_dir = os.path.join(output_dir, subject)
    
    # load connectome
    connect_matrix = load_matrix(connectivity_matrix_path)
    connect_matrix = drop_data_in_connect_matrix(connect_matrix)
    # load proteins concentration in brain regions
    t0_concentration = load_matrix(t0_concentration_path) 
    t1_concentration = load_matrix(t1_concentration_path)
        
    logging.info(f'Sum of t0 concentration: {np.sum(t0_concentration):.2f}')
    logging.info(f'Sum of t1 concentration: {np.sum(t1_concentration):.2f}')
    
    if (t0_concentration == t1_concentration).all():
        logging.info('Followup is the same as baseline. Subject skipped.')
        return
    
    try:
        simulation = MARsimulation(connect_matrix, t0_concentration, t1_concentration)
    except Exception as e:
        logging.error(f"Exception happened for \'simulation\' method of subject {subject}. Traceback:\n{e}\nTrying to plot the partial results...")

    t1_concentration_pred = simulation.run(norm_opt=2)
    t1_concentration_pred = drop_negative_predictions(t1_concentration_pred)
    rmse = calc_rmse(t1_concentration, t1_concentration_pred)
    corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
    if plot: visualize_terminal_state_comparison(t0_concentration, 
                                        t1_concentration_pred,
                                        t1_concentration,
                                        subject,
                                        rmse, 
                                        corr_coef)
    if save_results:
        save_terminal_concentration(subject_output_dir, t1_concentration_pred, 'MAR')
        save_coeff_matrix(subject_output_dir, simulation.coef_matrix)
    
    return simulation.coef_matrix
    
def parallel_training():   
    ''' 1st approach: train A matrix for each subject separately.
    The final matrix is an average matrix. '''

    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        patients = os.listdir(dataset_dir)
        results = executor.map(run_simulation, patients)
    
        avg_coeff_matrix = np.mean(results, axis=0)
        print(avg_coeff_matrix)
        
def sequential_training():
    patients = ['sub-CN4350']
    results = [run_simulation(pat) for pat in patients]
    
if __name__ == '__main__':
    sequential_training()