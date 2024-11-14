import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml

from tqdm import tqdm

PRED_FILE_NAME = "concentration_pred_SCM_t1_pred.csv"
SUBJECTS_PER_SEGMENT = 55
REGIONS_PER_SEGMENT = 45

def draw_regions_chart(data, output_path):
    mean_regions = np.mean(data, axis=0)
    std_regions = np.std(data, axis=0)
    min_regions = np.min(data, axis=0)
    max_regions = np.max(data, axis=0)

    x_pos = np.arange(len(mean_regions))

    plt.bar(x_pos, mean_regions, yerr=std_regions, capsize=5, alpha=0.7, color='skyblue', label='Mean')

    plt.scatter(x_pos, min_regions, color='red', zorder=5, label='Min')
    plt.scatter(x_pos, max_regions, color='green', zorder=5, label='Max')

    plt.xlabel("Column Index")
    plt.ylabel("Value")
    plt.title("Mean, Min, Max, and Standard Deviation of Regions")

    plt.savefig(output_path, bbox_inches='tight')

    plt.close()


def draw_segmented_saturation_change(base, sat_change, output_path):
    total_subject_segments_number = int(np.ceil(base.shape[0] / SUBJECTS_PER_SEGMENT))
    total_region_segments_number = int(np.ceil(base.shape[1] / REGIONS_PER_SEGMENT))
    for subj_seg_no in range(total_subject_segments_number):
        subj_seg_begin = subj_seg_no*SUBJECTS_PER_SEGMENT
        subj_seg_end = min((subj_seg_no+1)*SUBJECTS_PER_SEGMENT, base.shape[0])
        
        for reg_seg_no in range(total_region_segments_number):
            reg_seg_begin = reg_seg_no*REGIONS_PER_SEGMENT
            reg_seg_end = min((reg_seg_no+1)*REGIONS_PER_SEGMENT, base.shape[1])

            seg_base = base[subj_seg_begin:subj_seg_end, reg_seg_begin:reg_seg_end]
            set_sat_change = sat_change[subj_seg_begin:subj_seg_end, reg_seg_begin:reg_seg_end]

            x_pos = np.arange(seg_base.shape[1])+reg_seg_begin

            plt.figure(figsize=(12, 8))
            for i in range(seg_base.shape[0]):
                plt.scatter(x_pos, seg_base[i], c=set_sat_change[i], cmap='coolwarm', label=f'Row {i+1}', alpha=0.7)


            plt.colorbar(label='Saturation change intensity')

            plt.xlabel('Regions')
            plt.ylabel('Baseline-value')
            plt.title('Saturation change intensity from the baseline')

            plt.savefig(f"{output_path}_subj_{subj_seg_begin}-{subj_seg_end}_reg_{reg_seg_begin}-{reg_seg_end}.png", bbox_inches='tight')

            plt.close()
        




if __name__=="__main__":

    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    category = sys.argv[1] if len(sys.argv) > 1 else ''
    while category == '':
        try:
            category = input('Insert the category [ALL, AD, LMCI, MCI, EMCI, CN; default ALL]: ')
        except Exception as e:
            logging.info("Using default value")
            category = 'ALL'
        category = 'ALL' if category == '' else category

    
    
    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_res = config['paths']['dataset_dir'] + f'simulations/{category}/results/concentration/'
    if not os.path.exists(output_res):
        os.makedirs(output_res)

    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    

    all_baselines = np.empty((0, 166), float)
    all_followups = np.empty((0, 166), float)
    

    for subj, paths in tqdm(dataset.items()):
        baseline_conc = np.genfromtxt(paths['baseline'], delimiter=",")
        followup_conc = np.genfromtxt(paths['followup'], delimiter=",")

        # pred_conc_path = os.path.dirname(paths['baseline']) + f"/../../test/{PRED_FILE_NAME}"
        # if os.path.exists(pred_conc_path):
        #     pred_conc = np.genfromtxt(pred_conc_path, delimiter=",")
        # else:
        #     sys.exit(f"Error: Predicion file not found: {pred_conc_path}")

        
        all_baselines = np.concatenate((all_baselines, baseline_conc.reshape(1, -1)), axis=0)
        all_followups = np.concatenate((all_followups, followup_conc.reshape(1, -1)), axis=0)

    draw_regions_chart(all_baselines, output_res+"/baseline_region_distribution.png")
    draw_regions_chart(all_followups, output_res+"/followup_region_distribution.png")

    sat_change = all_followups / all_baselines - 1

    draw_segmented_saturation_change(all_baselines, sat_change, output_res+"/saturation_change")



