import os
import shutil

import SimpleITK as sitk
import numpy as np

def register(moving_img_path, fixed_img_path):
    # read img
    moving_img = sitk.ReadImage(moving_img_path, sitk.sitkFloat32)
    fixed_img = sitk.ReadImage(fixed_img_path, sitk.sitkFloat32)
    # normalization
    min_max_filter = sitk.MinimumMaximumImageFilter()
    min_max_filter.Execute(moving_img)
    maximum = min_max_filter.GetMaximum()
    moving_img = moving_img / maximum
    
    min_max_filter.Execute(fixed_img)
    maximum = min_max_filter.GetMaximum()
    fixed_img = fixed_img / maximum 
    #### registration process ####
    initial_transform = sitk.CenteredTransformInitializer(fixed_img, moving_img,
                                                          sitk.Similarity3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    # set similarity metric
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) # BSpline2
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.10)
    # set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    # set optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                      numberOfIterations=200,
                                                      convergenceMinimumValue=1e-6, 
                                                      convergenceWindowSize=50)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # calculate final transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed_img, sitk.sitkFloat32),
                                                  sitk.Cast(moving_img, sitk.sitkFloat32))
    print(moving_img_path, "\n", "Final metric value: {0}".format(registration_method.GetMetricValue()))
    # resample with final transformation
    # 0.0: default value
    moving_registered = sitk.Resample(moving_img, fixed_img, final_transform, 
                                      sitk.sitkLinear, 0.0, moving_img.GetPixelID())
    print("Optimizer stop condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
    print("Iteration: {0}".format(registration_method.GetOptimizerIteration()))
    ### DEGBUG
    # print('Moving image direction before: ', moving_image.GetDirection())
    # print('Moving image origin before: ', moving_image.GetOrigin())
    # print('Moving image direction after: ', moving_registered.GetDirection())
    # print('Moving image origin after: ', moving_registered.GetOrigin())
    # print('Fixed image idrection: ', fixed_image.GetDirection())
    # print('Fixed image origin:', fixed_image.GetOrigin())
    return moving_registered

def apply_transform(moving_img_path, fixed_img_path, transform_path):
    # read transform and img
    transform = sitk.ReadTransform(transform_path)
    moving_img = sitk.ReadImage(moving_img_path, sitk.sitkFloat32)
    fixed_img = sitk.ReadImage(fixed_img_path, sitk.sitkFloat32)
    registered = moving_img # initialization
    # tranform the mask
    if moving_img_path.endswith('mask.nii.gz'):
        registered = sitk.Resample(moving_img, fixed_img, transform, sitk.sitkNearestNeighbor, 0.0, moving_img.GetPixelID())
    # transform other images
    else:
        registered = sitk.Resample(moving_img, fixed_img, transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
        # normalization
        maximum = np.max(sitk.GetArrayFromImage(registered))
        registered = registered / maximum
    print('Successfully applied co-registration transform for ', moving_img_path)
    return registered
    
def register_to_T2(processed_cases_dir, patient_name):
    print("******** register structural MRIs to T2 ********")
    nifti_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                                 "2_nifti", patient_name)
    coreg2t2_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                                 "3_coreg2t2", patient_name)
    # to avoid generating duplicate nii files by dcm2niix
    os.makedirs(coreg2t2_dir, exist_ok=False)
    fixed_img_name = patient_name + "_T2.nii"
    for f in os.listdir(nifti_dir):
        if f.endswith("apt.nii") or f.endswith("apt_original.nii") or f.endswith("T2.nii"):
            shutil.copy(os.path.join(nifti_dir, f), os.path.join(coreg2t2_dir, f))
        if f.endswith("T1.nii") or f.endswith("T1c.nii") or f.endswith("Flair.nii"):
            result = register(os.path.join(nifti_dir, f), os.path.join(nifti_dir, fixed_img_name))
            sitk.WriteImage(result, os.path.join(coreg2t2_dir, f))
    
def register_to_APT(processed_cases_dir, patient_name, seq_dict):
    APTw_id = seq_dict["APTw_id"]
    print("******** register structural MRIs to APT ********")
    coreg2t2_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                                 "3_coreg2t2", patient_name)
    coreg2apt_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                                 "4_coreg2apt", patient_name)
    fixed_img_name = patient_name + "_" + str(APTw_id) + "_apt_original.nii"
    # to avoid generating duplicate nii files by dcm2niix
    os.makedirs(coreg2apt_dir, exist_ok=False)
    for f in os.listdir(coreg2t2_dir):
        if f.endswith("apt.nii") or f.endswith("apt_original.nii"):
            shutil.copy(os.path.join(coreg2t2_dir, f), os.path.join(coreg2apt_dir, f))
        if f.endswith("T2.nii") or f.endswith("T1.nii") or f.endswith("T1c.nii") or f.endswith("Flair.nii"):
            registered = apply_transform(os.path.join(coreg2t2_dir, f), os.path.join(coreg2t2_dir, fixed_img_name),
                                         os.path.join(coreg2t2_dir, "2apt.txt"))
            sitk.WriteImage(registered, os.path.join(coreg2apt_dir, f))
    # register MTRasym to APTw
    APTw = sitk.ReadImage(os.path.join(coreg2t2_dir, fixed_img_name), sitk.sitkFloat32)
    spacing = APTw.GetSpacing()
    origin = APTw.GetOrigin()
    direction = APTw.GetDirection()
    emr_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name+"_EMR")
    for f in os.listdir(emr_dir):
        if "apt" in f:
            img = sitk.ReadImage(os.path.join(emr_dir, f), sitk.sitkFloat32)
            img.SetSpacing(spacing)
            img.SetOrigin(origin)
            img.SetDirection(direction)
            sitk.WriteImage(img, os.path.join(coreg2apt_dir, f))
    return





      
        
        
        