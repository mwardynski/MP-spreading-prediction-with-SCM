"""
    SYNOPSIS
    python3 simulation_ESM.py <category> <cores> <mu> <lambdaD> <I_percentage> <NSteps>
"""


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

import networkx as nx

from utils_simplagion_MC import *
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

    def find_k2_k3(self, G):
        node_neighbors_dict = {}
        for n in G.nodes():
            node_neighbors_dict[n] = G[n]

        triangles_list = set()
        triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
        for triangle in triangles:
            triangles_list.add(tuple(sorted(triangle)))
        triangles_list = [list(tri) for tri in triangles_list]

        return node_neighbors_dict, triangles_list

    def import_connectome(self):

        G = nx.from_numpy_array(self.cm)
        node_neighbors_dict, triangles_list = self.find_k2_k3(G)

        N = len(node_neighbors_dict.keys())
        avg_k1 = 1.*sum([len(v) for v in node_neighbors_dict.values()])/N
        avg_k2 = 3.*len(triangles_list)/N

        return node_neighbors_dict, triangles_list, avg_k1, avg_k2
    
    def markovChain(self, beta, beta_D, mu, node_neighbors_dict, tri_neighbors_dict, NSteps, i0):
        N = len(node_neighbors_dict)
        p = self.t0_concentration
        p_new = np.copy(p)
        
        q = 1
        pTime = [np.mean(p)]
        for k in range(0,NSteps):
            for i in range(0,N):
                
                #Updating the q_i (infections) - d=1
                for j in node_neighbors_dict[i]:
                    wj = 1-node_neighbors_dict[i][j]['weight']
                    q *= (1.-beta*wj*p[j])
                    
                #Updating the q_i (infections) - d=2
                for j, k in tri_neighbors_dict[i]:
                    wj = 1-node_neighbors_dict[i][j]['weight']
                    wk = 1-node_neighbors_dict[i][k]['weight']
                    q *= (1.-beta_D*wj*p[j]*wk*p[k])
                
                #Updating the vector
                p_new[i] = (1-q)*(1-p[i]) + (1.-mu)*p[i]
                
                #Resetting the i-th parameters
                q = 1
                
            p = np.copy(p_new)
            pTime.append(np.mean(p))
        return np.mean(pTime[int(NSteps*0.8):]), p

    def simulation(self):
        self.cm = drop_data_in_connect_matrix(load_matrix(self.paths['CM']))

        node_neighbors_dict, triangles_list, avg_k1, avg_k2 = self.import_connectome()
        tri_neighbors_dict = get_tri_neighbors_dict(triangles_list)

        mu = 0.05
        lambda1s = np.linspace(0.0001,1.5,20)

        betas = 1.*(mu/avg_k1)*lambda1s

        t1_pred = None

        beta_D = 1.*(mu/avg_k2)*lambdaD
        i0 = I_percentage/100.
        rho_markov = []
        for beta in betas:
            rho_m, p = self.markovChain(beta, beta_D, mu, node_neighbors_dict, tri_neighbors_dict, NSteps, i0)
            rho_markov.append(rho_m)
        t1_pred = p


        return t1_pred

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

        beta_0 = float(sys.argv[3]) if len(sys.argv) > 3 else -1
    
    mu = float(sys.argv[3]) if len(sys.argv) > 3 else -1
    while mu < 0:
        try:
            mu = float(input('Insert the value for mu [default 0.05]: '))
        except Exception as e:
            logging.info('Using default value')
            mu = 0.05

    lambdaD = float(sys.argv[4]) if len(sys.argv) > 4 else -1
    while lambdaD < 0:
        try:
            lambdaD = float(input('Insert the value for lambdaD [default 2.5]: '))
        except Exception as e:
            logging.info('Using default value')
            lambdaD = 2.5

    I_percentage = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    while I_percentage < 0:
        try:
            I_percentage = int(input('Insert the value for I_percentage [default 1]: '))
        except Exception as e:
            logging.info('Using default value')
            I_percentage = 1

    NSteps = int(sys.argv[6]) if len(sys.argv) > 6 else -1
    while NSteps < 0:
        try:
            NSteps = int(input('Insert the value for NSteps [default 500]: '))
        except Exception as e:
            logging.info('Using default value')
            NSteps = 500

    
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
    out_file.write(f"mu: {mu}\n")
    out_file.write(f"lambdaD: {lambdaD}\n")
    out_file.write(f"I_percentage: {I_percentage}\n")
    out_file.write(f"NSteps: {NSteps}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string()+'\n')
    out_file.write(pt_subs.get_string())    
    out_file.close()
    logging.info('***********************')
    logging.info(f"Category: {category}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"mu: {mu}")
    logging.info(f"lambdaD: {lambdaD}")
    logging.info(f"NSteps: {NSteps}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"Subjects: {len(dataset.keys())}")
    logging.info(f"Total time (s): {format(total_time, '.2f')}")
    logging.info('***********************')
    logging.info(f"Results saved in {filename}")
    print(f"Results saved in {filename}")