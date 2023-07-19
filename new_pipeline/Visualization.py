import os
import cv2

import SimpleITK as sitk
import numpy as np

def convert_nifti_to_png_helper(input_dir, output_dir, f):
    img = sitk.ReadImage(os.path.join(input_dir, f))
    img_arr = sitk.GetArrayFromImage(img) # [15, 256, 256]
    img_arr = np.interp(img_arr, (img_arr.min(), img_arr.max()), (0., 1.))
    img_arr *= 255
    for i in range(img_arr.shape[0]):
        new_name = f.split(".")[0] + "_" + str(i+1) + ".png"
        cv2.imwrite(os.path.join(output_dir, new_name), np.flip(img_arr[i], 0))
        
def convert_nifti_to_png(processed_cases_dir, patient_name):
    print("******** convert nifti files to png files ********")
    coreg2apt_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                                 "4_coreg2apt", patient_name)
    png_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                           "5_png", patient_name)
    # to avoid generating duplicate png files
    os.makedirs(png_dir, exist_ok=False)
    for f in os.listdir(coreg2apt_dir):
        convert_nifti_to_png_helper(coreg2apt_dir, png_dir, f)

def read_lookup_table(lut_dir):
    lookup_table = []
    with open(lut_dir, "r") as lut_file:
        lut_lines = lut_file.read().split('\n')
        for line in lut_lines:
            if len(line) > 0:
                line_nums = [int(i) for i in line.split('\t')]
                lookup_table.append(line_nums) # 4 columns: Gray - R - G - B
    lookup_table = np.array(lookup_table) 
    return lookup_table

def gray_to_idl(gray_img, lookup_table):
    # print("Min:", np.min(gray_img))
    # print("Max:", np.max(gray_img))
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            g_val = gray_img[i, j, 2]
            gray_img[i, j, 0] = lookup_table[g_val, 3] # blue channel
            gray_img[i, j, 1] = lookup_table[g_val, 2] # green channel
            gray_img[i, j, 2] = lookup_table[g_val, 1]
    return gray_img

def sort_image_name(img_name_list):
    """Input: Python list of image names
    - name format: xxxxxxxxx_(modality)_(slice number).png
    Output: list of image names sorted by slice_number"""
    img_slice_list = []
    for name in img_name_list:
        nameparts = name.split('_')
        img_slice = int(nameparts[-1].split('.')[0])
        img_slice_list.append(img_slice)
    zipped_name_slice = zip(img_slice_list, img_name_list)
    sorted_name_list = [x for _, x in sorted(zipped_name_slice)]
    return sorted_name_list

def generate_tiles_single(n_cols, patient_name, seq, input_dir, output_dir, lookup_table=None):
    h_patch_list = []
    n_slices = 15
    for i in range(n_slices):
        if i % n_cols == 0:
            h_patch = []
        img_path = os.path.join(input_dir, patient_name+"_"+seq+"_"+str(i+1)+".png")
        img = cv2.imread(img_path)
        if lookup_table is not None:
            img = gray_to_idl(img, lookup_table)
        h_patch.append(img)
        if i % n_cols == n_cols-1:
            h_patch_list.append(np.concatenate(h_patch, axis=1))
    final_img = np.concatenate(h_patch_list, axis = 0)
    cv2.imwrite(os.path.join(output_dir, patient_name+"_"+seq+".png"), final_img)
    print(seq, "single tile:", final_img.shape)
    return final_img 

def generate_tiles_multiple(order_list, n_rows, patient_name, seq, input_dir, output_dir, lookup_table):
    n_slices = 15
    for i in range(n_slices):
        if i % n_rows == 0:
            h_patch_list = [] # this will have shape (n_rows*height, n_modalities*width, 3)
        row_patch_list = []
        for modality in order_list:
            img_path = os.path.join(input_dir, patient_name+"_"+modality+"_"+str(i+1)+".png")
            img = cv2.imread(img_path)
            if "apt" in modality:
                img = gray_to_idl(img, lookup_table)
            row_patch_list.append(img)
        h_patch_list.append(np.concatenate(row_patch_list,axis = 1))
        if i % n_rows == n_rows - 1:
            final_img = np.concatenate(h_patch_list, axis = 0)
            cv2.imwrite(os.path.join(output_dir, patient_name+"_"+str(i-n_rows+2)+"_"+str(i+1)+".png"),
                        final_img)
            print("multiple tile:", final_img.shape)
    return
          
def generate_png_tiles(processed_cases_dir, patient_name, lut_dir, seq_dict):
    MTR_1p5uT_id = seq_dict["MTR_1p5uT_id"]
    MTR_2uT_id = seq_dict["MTR_2uT_id"]
    APTw_id = seq_dict["APTw_id"]
    APTw_cs4_id = seq_dict["APTw_cs4_id"]
    # get order list
    order_list = ["T2", "Flair", "T1", "T1c"]
    # order_list = ["Flair", "T1"]
    if APTw_id > 0:
        order_list.append(str(APTw_id)+"_apt")
    if MTR_2uT_id > 0:
        order_list.append(str(MTR_2uT_id)+"_apt")
    elif MTR_1p5uT_id > 0:
        order_list.append(str(MTR_1p5uT_id)+"_apt")
    # get single list
    single_list = ["T1c"]
    # single_list = []
    if APTw_id > 0:
        single_list.append(str(APTw_id)+"_apt")
    if APTw_cs4_id > 0:
        single_list.append(str(APTw_cs4_id)+"_apt")
    if MTR_1p5uT_id > 0:
        single_list.append(str(MTR_1p5uT_id)+"_apt")
    if MTR_2uT_id > 0:
        single_list.append(str(MTR_2uT_id)+"_apt")
    # 4 columns: Gray - R - G - B
    lookup_table = read_lookup_table(lut_dir)
    print("******** convert nifti files to png files ********")
    png_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                                 "5_png", patient_name)
    png_tile_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                           "6_png_tiles", patient_name)
    # to avoid generating duplicate png tile files
    os.makedirs(png_tile_dir, exist_ok=False)
    # generate single tiles
    for seq in single_list:
        if "apt" in seq:
            # 5 cols * 3 rows
            generate_tiles_single(5, patient_name, seq, png_dir, png_tile_dir, lookup_table)
        else:
            generate_tiles_single(5, patient_name, seq, png_dir, png_tile_dir, None)
    # generate multiple tiles
    generate_tiles_multiple(order_list, 3, patient_name, seq, png_dir, png_tile_dir, lookup_table)
    
      
        
        
        