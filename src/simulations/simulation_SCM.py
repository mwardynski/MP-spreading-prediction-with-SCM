from datetime import datetime
import json
from threading import Thread, Lock
from multiprocessing import cpu_count
import os
import logging
import sys
from time import time, sleep
import warnings
from prettytable import PrettyTable
import yaml
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr as pearson_corr_coef
from sklearn.metrics import mean_squared_error
import re

from utils_vis import *
from utils import *

sim_name = 'SCM'
np.seterr(all = 'raise')
date = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_{sim_name}_{date}.log")
digits = 4

class SCM(Thread):
    
    def __init__(self, paths, subj, n_regions = 166):
        Thread.__init__(self)
        self.paths = paths 
        self.subj = subj
        self.n_regions = n_regions

    def simulation(self):
        return self.t0_concentration

    def run(self):
        logging.info(f"Starting simulation for subject {self.subj}")
        if not os.path.exists(self.subj + 'test/'):
                os.makedirs(self.subj + 'test/')
        try:
            self.cm = drop_data_in_connect_matrix(load_matrix(self.paths['CM']))

            self.t0_concentration = load_matrix(self.paths['baseline'])
            self.t0 = np.copy(self.t0_concentration)
            self.t1_concentration = load_matrix(self.paths['followup'])
        except Exception as e:
            logging.error(f'Error appening while loading data of subject {self.subj}. Traceback: {e}')
            return

        try:
            self.t1_concentration_pred = self.simulation()

            if np.isnan(self.t1_concentration_pred).any() or np.isinf(self.t1_concentration_pred).any(): raise Exception("Discarding prediction")
        except Exception as e:
            logging.error(f'Error during simulation for subject {self.subj}. Traceback: {e}')
            return
        
        try:
            mse = mean_squared_error(self.t1_concentration, self.t1_concentration_pred)
            pcc = pearson_corr_coef(self.t1_concentration, self.t1_concentration_pred)[0]
            if np.isnan(mse) or np.isinf(mse): raise Exception("Invalid value of MSE")
            if np.isnan(pcc): raise Exception("Invalid value of PCC")
        except Exception as e:
            logging.error(f'Error appening during computation of MSE and PCC for subject {self.subj}. Traceback: {e}')
            return
        
        
        reg_err = np.abs(self.t1_concentration_pred - self.t1_concentration)
        
        lock.acquire()
        # TODO: uncomment the lower line
        save_prediction_plot(self.t0_concentration, self.t1_concentration_pred, self.t1_concentration, self.subj, self.subj + 'test/' + sim_name + '_' + date + '.png', mse, pcc)
        logging.info(f"Saving prediction in {self.subj + 'test/' + sim_name + '_' + date + '.png'}")
        total_mse[self.subj] = mse
        total_pcc[self.subj] = pcc
        total_reg_err[self.subj] = reg_err
        pt_subs.add_row([self.subj, round(mse,digits), round(pcc,digits)])
        lock.release()
            
        return

if __name__=="__main__":

    ### INPUT ###

    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if os.getcwd().endswith('simulations'):
        os.chdir(os.getcwd()+'/../../..')
    category = sys.argv[1] if len(sys.argv) > 1 else ''
    while category == '':
        try:
            category = input('Insert the category [ALL, AD, LMCI, MCI, EMCI, CN; default ALL]: ')
        except Exception as e:
            logging.info("Using default value")
            category = 'ALL'
        category = 'ALL' if category == '' else category

    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_res = config['paths']['dataset_dir'] + f'simulations/{category}/results/'
    output_mat = config['paths']['dataset_dir'] + f'simulations/{category}/matrices/'
    if not os.path.exists(output_res):
        os.makedirs(output_res)
    if not os.path.exists(output_mat):
        os.makedirs(output_mat)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    num_cores = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    while num_cores < 1:
        try:
            num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
        except Exception as e:
            num_cores = cpu_count()
            logging.info(f"{num_cores} cores available")

    
    ### SIMULATIONS ###

    pt_avg = PrettyTable()
    pt_avg.field_names = ["CG", "Avg MSE", "SD MSE", "Avg Pearson", "SD Pearson"]
    
    pt_subs = PrettyTable()
    pt_subs.field_names = ["ID", "MSE", "Pearson"]
    pt_subs.sortby = "ID" # Set the table always sorted by patient ID

    total_mse = {}
    total_pcc = {}
    total_reg_err = {}
    
    total_time = time()

    lock = Lock()
    
    works = []
    for subj, paths in tqdm(dataset.items()):
        works.append(SCM(paths, subj))
        works[-1].start()
        while len (works) >= num_cores:
            for w in works:
                if not w.is_alive():
                    works.remove(w)
            
    for w in works:
        w.join()
        works.remove(w)
        
        
    total_time = time() - total_time
    sleep(1)   
    ### OUTPUT ###
    categories = ['AD', 'LMCI', 'MCI', 'EMCI', 'CN', 'Decreasing', 'Increasing']
    
    
    for c in categories:
        cat_reg_err = []
        cat_total_mse = []
        cat_total_pcc = []
        for sub in total_reg_err.keys():
            if re.match(rf".*sub-{c}.*", sub):
                cat_reg_err.append(total_reg_err[sub])
                cat_total_mse.append(total_mse[sub])
                cat_total_pcc.append(total_pcc[sub])

        if len(cat_reg_err) == 0:
            continue
        avg_reg_err = np.mean(cat_reg_err, axis=0)
        avg_reg_err_filename = output_res +f'{sim_name}_region_{c}_{date}.png'
        save_avg_regional_errors(avg_reg_err, avg_reg_err_filename)
        np.savetxt(f"{output_mat}{sim_name}_{c}_regions_{date}.csv", avg_reg_err, delimiter=',')
        avg_mse = np.mean(cat_total_mse, axis=0)
        std_mse = np.std(cat_total_mse, axis=0)
        avg_pcc = np.mean(cat_total_pcc, axis=0)
        std_pcc = np.std(cat_total_pcc, axis=0)
        
        pt_avg.add_row([c, round(avg_mse, digits), round(std_mse, 2), round(avg_pcc, digits), round(std_pcc, 2)])
    
    if category not in categories:
        pt_avg.add_row([category, round(np.mean(list(total_mse.values())), digits), round(np.std(list(total_mse.values())), 2), round(np.mean(list(total_pcc.values())), digits), round(np.std(list(total_pcc.values())), 2)])
        avg_reg_err = np.mean(list(total_reg_err.values()), axis=0)
        avg_reg_err_filename = output_res +f'{sim_name}_region_{category}_{date}.png'
        save_avg_regional_errors(avg_reg_err, avg_reg_err_filename)
        np.savetxt(f"{output_mat}{sim_name}_{category}_regions_{date}.csv", avg_reg_err, delimiter=',')

    filename = f"{output_res}{sim_name}_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string()+'\n')
    out_file.write(pt_subs.get_string())    
    out_file.close()
    logging.info('***********************')
    logging.info(f"Category: {category}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"Subjects: {len(dataset.keys())}")
    logging.info(f"Total time (s): {format(total_time, '.2f')}")
    logging.info('***********************')
    logging.info(f"Results saved in {filename}")
    print(f"Results saved in {filename}")