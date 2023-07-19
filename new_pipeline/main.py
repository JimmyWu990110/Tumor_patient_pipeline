import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import SimpleITK as sitk

from DataPreprocessing import formalize_par_rec, query_par_rec_save_info, convert_selected_par_rec_to_nifti, process_raw_nifti, sepcify_sequence_ids
from Registration import register_to_T2, register_to_APT
from Mapping import get_nifti_for_fitting, get_T1_T2_map
from MTRasymCalculation import cal_MTRasym
from Visualization import convert_nifti_to_png, generate_png_tiles

new_cases_dir = r"C:\Users\jwu191\Desktop\new_cases"
processed_cases_dir = r"C:\Users\jwu191\Desktop\processed_cases"
patient_name = "APT_518"
lut_dir = r"C:\Users\jwu191\Desktop\new_pipeline\idl_rainbow.lut"

# formalize_par_rec(new_cases_dir, patient_name)
# query_par_rec_save_info(new_cases_dir, processed_cases_dir, patient_name)
# convert_selected_par_rec_to_nifti(new_cases_dir, processed_cases_dir, patient_name)
# process_raw_nifti(processed_cases_dir, patient_name)
# register_to_T2(processed_cases_dir, patient_name)
"""
STOP HERE: Fine tune with ITK-Snap and get 2apt.txt, 
sepcify seuqnce ids, if not exist, use -1
"""
seq_dict = sepcify_sequence_ids(MTR_1p5uT_id=5, WASSR_EMR_id=6, MTR_2uT_id=7, 
                                APTw_id=10, APTw_cs4_id=11)
get_nifti_for_fitting(new_cases_dir, processed_cases_dir, patient_name)
cal_MTRasym(processed_cases_dir, patient_name, seq_dict)
register_to_APT(processed_cases_dir, patient_name, seq_dict)
convert_nifti_to_png(processed_cases_dir, patient_name)
generate_png_tiles(processed_cases_dir, patient_name, lut_dir, seq_dict)


