Conventional (structural sequences and APTw, MTR) pipeline for processing a patient

In main.py, change these parameters:
  -new_cases_dir: the folder storing the par-rec files for all patients
  -processed_cases_dir: the folder storing the processed data for all patients
  -patient_name: the patient you're processing (eg. "APT_505")
  -lut_dir: the path of the color bar file (idl_rainbow.lut) 

After that, run these function:
  formalize_par_rec(new_cases_dir, patient_name)
  query_par_rec_save_info(new_cases_dir, processed_cases_dir, patient_name)
  convert_selected_par_rec_to_nifti(new_cases_dir, processed_cases_dir, patient_name)
  process_raw_nifti(processed_cases_dir, patient_name)
  register_to_T2(processed_cases_dir, patient_name)

A folder "patient_name_all" will be created under the processed_cases_dir, it consists of 
"1_raw_output", "2_nifti", "3_coreg2t2" and an excel file listing all sequence info of this patient
Open "3_coreg2t2" folder and fine-tune the registration (Flair, T1w, T1c -> T2w, level=2) using ITK-SNAP
Get transform (T2w -> APTw_original, level=1) and save it as "2apt.txt" in "3_coreg2t2" folder

Before continuing, open the excel file and find the sequence ids for EMR_1.5uT, WASSR, EMR_2uT, APTw and APTw_CS4
eg. for APT_505 it's 6, 7, 8, 11 and 12, sepcify these ids below (if not exist, use -1) and run the functions:
  seq_dict = sepcify_sequence_ids(MTR_1p5uT_id=6, WASSR_EMR_id=7, MTR_2uT_id=8, 
                                  APTw_id=11, APTw_cs4_id=12)
  get_nifti_for_fitting(processed_cases_dir, mapping_dir, patient_name)
  cal_MTRasym(processed_cases_dir, mapping_dir, patient_name, seq_dict)
  register_to_APT(processed_cases_dir, patient_name, seq_dict)
  convert_nifti_to_png(processed_cases_dir, patient_name)
  generate_png_tiles(processed_cases_dir, patient_name, lut_dir, seq_dict)

Then you will get "4_coreg2apt", "5_png" and "6_png_tiles" foldesr containing all the sequences in nifti, png and png tile format
Copy and paste png_tiles to PPT for report







