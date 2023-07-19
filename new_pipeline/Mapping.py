import numpy as np
import math
from scipy.optimize import curve_fit
from scipy import stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import pandas as pd

def CovertPARToNIFI(base_dir, out_dir, df):
    os.makedirs(out_dir, exist_ok=True)
    for seq_idx in range(len(df)):
        patient_id = df.iloc[seq_idx]['Patient_id']
        protocol = df.iloc[seq_idx]['Protocol']
        sequence_id = df.iloc[seq_idx]['Sequence_id']
        if 'Look Locker' in protocol:
            print("T1: ", protocol)
            image_name = '%f_%p_%t_%s'
            target_path = os.path.join(base_dir, patient_id, sequence_id + '.rec')
            output_path = os.path.join(out_dir, 'T1-nifti')
            os.makedirs(output_path, exist_ok=True)
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_path + ' ' + target_path)
        elif 'T2_ME' in protocol:
            print("T2: ", protocol)
            image_name = '%f_%p_%t_%s'
            target_path = os.path.join(base_dir, patient_id, sequence_id + '.rec')
            output_path = os.path.join(out_dir, 'T2-nifti')
            os.makedirs(output_path, exist_ok=True)
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_path + ' ' + target_path)
        elif 'Zspec' in protocol or 'Z_single' in protocol or 'EMR' in protocol:
            print("EMR: ", protocol)
            image_name = '%f_%p_%t_%s'
            target_path = os.path.join(base_dir, patient_id, sequence_id + '.rec')
            output_path = os.path.join(out_dir, 'EMR-nifti')
            os.makedirs(output_path, exist_ok=True)
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_path + ' ' + target_path)  
        elif 'WASSR' in protocol:
            print("WASSR: ", protocol)
            image_name = '%f_%p_%t_%s'
            target_path = os.path.join(base_dir, patient_id, sequence_id + '.rec')
            output_path = os.path.join(out_dir, 'WASSR-nifti')
            os.makedirs(output_path, exist_ok=True)
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_path + ' ' + target_path) 

def ReadData_T1(base_dir):
    nii_files = []
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if name.endswith('.nii'):
                nii_files.append(name)
    if nii_files == []:
        print('No files for reading!')
        return
    else:
        all_images = []
        all_t = []
        for name in nii_files:
            name_parts = name.split('_')
            time = name_parts[-1].split('.')[0][1:]
            all_t.append(int(time))
            image = sitk.ReadImage(os.path.join(base_dir, name))
            image_array = sitk.GetArrayFromImage(image)
            all_images.append(image_array)
        all_images = [all_images for _, all_images in sorted(zip(all_t, all_images))]
        all_t = sorted(all_t)
        all_t = np.array(all_t) * 10 ** -3
        all_images = np.array(all_images)
        # Get image physical information for later use
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        imginfo = [spacing, origin, direction]
        print("T1 time:", all_t)
        print("T1 image shape:", all_images.shape)
        return all_t, all_images, imginfo

def ReadData_T2(base_dir):
    nii_files = []
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if name.endswith('.nii'):
                nii_files.append(name)
    if nii_files == []:
        print('No files for reading!')
        return
    else:
        all_images = []
        all_t = []
        for name in nii_files:
            name_parts = name.split('_')
            time = name_parts[-1].split('.')[0][1:]
            try:
                all_t.append(int(time))
            except ValueError:
                continue
            image = sitk.ReadImage(os.path.join(base_dir, name))
            image_array = sitk.GetArrayFromImage(image)
            all_images.append(image_array)
        # Get image physical information for later use
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        imginfo = [spacing, origin, direction]
        # Sort data based on time echo
        all_images = [all_images for _, all_images in sorted(zip(all_t, all_images))]
        all_t = sorted(all_t)
        all_t = np.array(all_t) * 0.03
        all_images = np.array(all_images)       
        print("T2 time:", all_t)
        print("T2 image shape:", all_images.shape)
        return all_t, all_images, imginfo

def removebackground(pid, image):
    processedpath = r"C:\Users\jwu191\Desktop\output"
    apt_mask_name = pid + '_11_apt.nii'
    apt_mask = sitk.ReadImage(os.path.join(processedpath, pid+'_all', pid, '2_nifti', pid, apt_mask_name))
    apt_mask = sitk.GetArrayFromImage(apt_mask)
    apt_mask = (apt_mask == -5)
    image_new = image
    image_new[apt_mask] = 0
    return image_new

def func(t, a, b, t1_star):
    return a - b * np.exp(-t / t1_star)

def calculate_t1(a, b, t1_star):
    return t1_star * (b / a - 1)
    
def t1_map_one_slice(all_images, slice_idx, all_t, flag=0):
    # all_images (n, s, x, y)
    # flag = 0: minimal points keep the same, flag = 1:mininal points become negative
    x_data = all_t
    img = all_images[:, slice_idx, :, :]
    res = np.zeros(img[0].shape)
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            try:
                if np.sum(img[:, x, y]) == 0:
                    continue
                else:
                    num_points = np.argmin(img[:, x, y])
                    y_data = np.array(img[:, x, y])
                    y_data[:num_points + flag] = -img[:num_points + flag, x, y]
                    popt, pcov = curve_fit(func, x_data, y_data, method='lm', p0=[3e+05, 8e+05, 1],
                                           maxfev=5000)
                    res[x][y] = calculate_t1(*popt)
            except RuntimeError:
                res[x][y] = 0
    return res

def t1_map_3d(all_images, all_t, flag=0):
    res_3d = []
    for sl in range(all_images.shape[1]):
        print("slice", sl, "processing...")
        res_sl = t1_map_one_slice(all_images, sl, all_t, flag)
        res_3d.append(res_sl)
    return np.array(res_3d)

def T1mapping(base_dir, out_dir, pid):
    all_t, all_images, imginfo = ReadData_T1(base_dir)
    t1_map_all = t1_map_3d(all_images, all_t, 1)

    t1_map_all[t1_map_all < 0] = 0
    t1_map_all[t1_map_all > 3] = 3

    # t1_map_all = removebackground(pid, t1_map_all)

    t1_map_img = sitk.GetImageFromArray(t1_map_all)
    t1_map_img.SetSpacing(imginfo[0])
    t1_map_img.SetOrigin(imginfo[1])
    t1_map_img.SetDirection(imginfo[2])
    save_name = 't1_map' + '.nii'

    sitk.WriteImage(t1_map_img, os.path.join(out_dir, save_name))
    
def func_t2_fitting(t, a, t2):
    return a * np.exp(-t / t2)

def t2_map_one_slice(all_images, sl, all_t, num_points):
    x_data = all_t[-num_points:]
    y_data = np.log(all_images[-num_points:, sl, :, :])
    res = np.zeros(y_data[0].shape)
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data[:, x, y])
            res[x][y] = -1 / slope

    return res

def t2_map_3d(all_images, all_t, num_points):
    res_3d = []
    for sl in range(all_images.shape[1]):
        print("slice", sl, "processing...")
        res_sl = t2_map_one_slice(all_images, sl, all_t, num_points)
        res_3d.append(res_sl)
    return np.array(res_3d)

def T2mapping(base_dir, out_dir, pid):
    all_t, all_images, imginfo = ReadData_T2(base_dir)
    num_points = len(all_t)
    t2_map_all = t2_map_3d(all_images, all_t, num_points)
    t2_map_5_copy = np.copy(t2_map_all)
    t2_map_5_copy[np.isnan(t2_map_5_copy) == True] = 0
    t2_map_5_copy[t2_map_5_copy < 0] = 0
    t2_map_5_copy[t2_map_5_copy > 0.4] = 0.4
    
    # t2_map_5_copy = removebackground(pid, t2_map_5_copy)

    t2_map_img = sitk.GetImageFromArray(t2_map_5_copy)
    t2_map_img.SetSpacing(imginfo[0])
    t2_map_img.SetOrigin(imginfo[1])
    t2_map_img.SetDirection(imginfo[2])
    save_name = 't2_map' + '.nii'
    sitk.WriteImage(t2_map_img, os.path.join(out_dir, save_name))
    

def get_nifti_for_fitting(new_cases_dir, processed_cases_dir, patient_name):
    out_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name+"_mapping")
    os.makedirs(out_dir, exist_ok=False)       
    df = pd.read_excel(os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                                  "all_scan_info.xlsx"))     
    CovertPARToNIFI(new_cases_dir, out_dir, df) 

def get_T1_T2_map(new_cases_dir, processed_cases_dir, patient_name):
    out_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name+"_mapping")
    T1mapping(os.path.join(out_dir, "T1-nifti"), out_dir, patient_name)
    T2mapping(os.path.join(out_dir, "T2-nifti"), out_dir, patient_name) 




       
        
        