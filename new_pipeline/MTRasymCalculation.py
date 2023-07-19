import os
import cv2
import math

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import interpolate


def read_WASSR(data_dir, seq_id):
    print("******** read WASSR ********")
    wassr_dir = os.path.join(data_dir, "WASSR-nifti")
    filenames = os.listdir(wassr_dir)
    suffix = "_" + str(seq_id) + ".nii"
    target_path = ""
    for f in filenames:
        if "WASSR" in f and suffix in f:
            target_path = os.path.join(wassr_dir, f)
            break
    if target_path == "":
        raise Exception("WASSR sequence id " + str(seq_id) + " not found!")
    img = sitk.ReadImage(target_path, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)
    print(target_path + " has been read.")
    return arr
 
def read_EMR(data_dir, seq_id):
    print("******** read EMR ********")
    emr_dir = os.path.join(data_dir, "EMR-nifti")
    filenames = os.listdir(emr_dir)
    suffix = "_" + str(seq_id) + ".nii"
    target_path = ""
    for f in filenames:
        if "EMR" in f and suffix in f:
            target_path = os.path.join(emr_dir, f)
            break
    if target_path == "":
        raise Exception("EMR sequence id " + str(seq_id) + " not found!")
    img = sitk.ReadImage(target_path, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)
    print(target_path + " has been read.")
    return arr

def get_WASSR_26_offsets():
    # hertz
    offset = [np.inf, 0, 14, -14, 28, -28, 42,-42, 56, -56, 70, -70, 84, -84, 98, -98, 
              112, -112, 126, -126, 140, -140, 154, -154, 168, -168]
    return np.array(offset)

def get_EMR_24_offsets():
    # ppm
    offset = [np.inf, 80, 60, 40, 30, 20, 10, 8, 4, -4, 3.5, -3.5, 3.5, -3.5, 
              3, -3, 2.5, -2.5, 2, -2, 2, -2, 1.5, -1.5]
    return np.array(offset)

def get_zero_mask(emr_data, threshold):
    offset = get_EMR_24_offsets()
    M0 = emr_data[0] # [15, 256, 256]
    zero_mask = np.ones((M0.shape[0], M0.shape[1], M0.shape[2]))
    for i in range(M0.shape[0]):
        single_slice_smoothed = cv2.blur(M0[i], (21, 21))
        single_slice_mask = np.ones((M0.shape[1], M0.shape[2]))
        single_slice_mask[np.where(single_slice_smoothed < threshold)] = 0
        zero_mask[i] = single_slice_mask
    return zero_mask
    
def cal_MTR_asym(emr_data, zero_mask):
    offset = get_EMR_24_offsets()
    M0 = emr_data[0]
    downfield_index = np.where(offset == 3.5)[0] # 3.5 ppm
    upfield_index = np.where(offset == -3.5)[0] # -3.5 ppm
    downfield_imgs = emr_data[downfield_index]
    downfield_avg = np.mean(downfield_imgs, axis=0)
    upfield_imgs = emr_data[upfield_index]
    upfield_avg = np.mean(upfield_imgs, axis=0)
    # (Img_-3.5 - Img_3.5) / M0
    diff = upfield_avg - downfield_avg # [15, 256, 256]
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            for k in range(diff.shape[2]):
                if zero_mask[i][j][k] == 0 or M0[i][j][k] == 0:
                    diff[i][j][k] = -5
                else:
                    diff[i][j][k] /= M0[i][j][k] 
    diff *= 100 # in percent 
    # normalize to [-5, 5]
    diff[np.where(diff > 5)] = 5
    diff[np.where(diff < -5)] = -5     
    print(np.min(diff), np.max(diff))
    # plt.hist(diff.flatten(), bins=100)
    # plt.show()
    return diff

def polyFitOnePixel(Zspec, offset):
    if np.max(Zspec) < 1000:
        return 0
    sort_index = np.argsort(offset)
    offset_sorted = np.sort(offset)
    y_sorted = Zspec[sort_index]
    x_upsampled = np.arange(-168, 169, 1) # [-168, -167, ... 167, 168]
    paras = np.polyfit(offset_sorted, y_sorted, deg=12)
    p = np.poly1d(paras)
    index = np.argmin(p(x_upsampled))
    # plt.scatter(offset_sorted, y_sorted)
    # plt.plot(x_upsampled, p(x_upsampled))
    # plt.show()
    # print(x_upsampled[index])
    return x_upsampled[index]
    
def cal_B0_shift_map(wassr):
    offset = get_WASSR_26_offsets()[1:]
    M0 = wassr[0] # [15, 128, 128]
    B0_shift_map = np.zeros((M0.shape[0], M0.shape[1], M0.shape[2]))
    for i in range(M0.shape[0]):
        for j in range(M0.shape[1]):
            for k in range(M0.shape[2]):                  
                Zspec = wassr[1:, i, j, k]
                B0_shift_map[i][j][k] = polyFitOnePixel(Zspec, offset) 
    print(np.min(B0_shift_map), np.max(B0_shift_map))
    return B0_shift_map

def B0CorrOnePixel(Zspec, offset, B0_shift):
    M0_index = np.where(offset == np.inf)
    M0 = Zspec[M0_index]
    Zspec /= M0
    offset *= 128
    pos_384_hz = np.mean(Zspec[np.where(offset == 384)[0]])
    pos_448_hz = np.mean(Zspec[np.where(offset == 448)[0]])
    pos_512_hz = np.mean(Zspec[np.where(offset == 512)[0]])
    neg_384_hz = np.mean(Zspec[np.where(offset == -384)[0]])
    neg_448_hz = np.mean(Zspec[np.where(offset == -448)[0]])
    neg_512_hz = np.mean(Zspec[np.where(offset == -512)[0]])
    x_pos = np.array([384, 448, 512])
    y_pos = np.array([pos_384_hz, pos_448_hz, pos_512_hz])
    x_interp_pos = np.arange(448-168, 448+168+1, 1) # [280, 281, ... 615, 616]
    func_pos = interpolate.interp1d(x_pos, y_pos, "linear", fill_value="extrapolate")
    y_interp_pos = func_pos(x_interp_pos)
    pos_448_hz_corrected = y_interp_pos[np.where(x_interp_pos == 448+B0_shift)][0]
    # plt.plot(x_interp_pos, y_interp_pos)
    # plt.scatter(x_pos, y_pos)
    # plt.show()  
    x_neg = np.array([-512, -448, -384])
    y_neg = np.array([neg_512_hz, neg_448_hz, neg_384_hz])
    x_interp_neg = np.arange(-448-168, -448+168+1, 1) # [-616, -615, ... -281, -280]
    func_neg = interpolate.interp1d(x_neg, y_neg, "linear", fill_value="extrapolate")
    y_interp_neg = func_neg(x_interp_neg)
    neg_448_hz_corrected = y_interp_neg[np.where(x_interp_neg == -448+B0_shift)][0]
    # plt.plot(x_interp_neg, y_interp_neg)
    # plt.scatter(x_neg, y_neg)
    # plt.show()   
    # print("MTR asym before correction:", neg_448_hz - pos_448_hz)
    # print("MTR asym after correction:", neg_448_hz_corrected - pos_448_hz_corrected)
    return 100 * (neg_448_hz_corrected - pos_448_hz_corrected)

def correct_B0(emr_data, B0_shift_map, zero_mask):
    M0 = emr_data[0] # [15, 256, 256]
    MTR_asym = np.zeros((M0.shape[0], M0.shape[1], M0.shape[2]))
    for i in range(M0.shape[0]):
        print("slice", i+1, "...")
        for j in range(M0.shape[1]):
            for k in range(M0.shape[2]):
                if zero_mask[i][j][k] == 0 or M0[i][j][k] == 0:
                    MTR_asym[i][j][k] = -5
                else:
                    Zspec = emr_data[:, i, j, k]
                    offset = get_EMR_24_offsets()
                    B0_shift = B0_shift_map[i][int(j/2)][int(k/2)]
                    MTR_asym[i][j][k] = B0CorrOnePixel(Zspec, offset, B0_shift)
    MTR_asym[np.where(MTR_asym > 5)] = 5
    MTR_asym[np.where(MTR_asym < -5)] = -5     
    # print(np.min(MTR_asym), np.max(MTR_asym))
    return MTR_asym

def cal_MTRasym(processed_cases_dir, patient_name, seq_dict):
    mapping_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name+"_mapping")
    emr_1p5uT_id = seq_dict["MTR_1p5uT_id"]
    wassr_emr_id = seq_dict["WASSR_EMR_id"]
    emr_2uT_id = seq_dict["MTR_2uT_id"]
    if wassr_emr_id <= 0:
        return
    wassr_emr = read_WASSR(mapping_dir, wassr_emr_id)
    emr_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name+"_EMR")
    os.makedirs(emr_dir, exist_ok=True)
    B0_shift_map = cal_B0_shift_map(wassr_emr)
    B0_shift_img = sitk.GetImageFromArray(B0_shift_map)
    sitk.WriteImage(B0_shift_img, os.path.join(emr_dir, "B0_shift_map.nii")) 
    if emr_1p5uT_id > 0:
        emr_1p5uT = read_EMR(mapping_dir, emr_1p5uT_id)
        zero_mask = get_zero_mask(emr_1p5uT, threshold=2e6)  
        img_arr = correct_B0(emr_1p5uT, B0_shift_map, zero_mask) 
        img = sitk.GetImageFromArray(img_arr)
        sitk.WriteImage(img, os.path.join(emr_dir, patient_name+"_"+str(emr_1p5uT_id)+"_apt.nii")) 
    if emr_2uT_id > 0:
        emr_2uT = read_EMR(mapping_dir, emr_2uT_id)
        zero_mask = get_zero_mask(emr_2uT, threshold=2e6)  
        img_arr = correct_B0(emr_2uT, B0_shift_map, zero_mask) 
        img = sitk.GetImageFromArray(img_arr)
        sitk.WriteImage(img, os.path.join(emr_dir, patient_name+"_"+str(emr_2uT_id)+"_apt.nii")) 
    return

 
    
 
    
 
    
 
    
 
    
        