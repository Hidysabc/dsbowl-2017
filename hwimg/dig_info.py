#!/usr/bin/env python

import numpy as np
import os
import dicom
import pandas as pd
import sys
import argparse


def scan_dicom(patient_id, patients_folder):
    path = os.path.join(patients_folder, patient_id)
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    img = np.stack([s.pixel_array for s in slices])
    img_z = img.shape[0]
    img_x = img.shape[1]
    img_y = img.shape[2]
    spacing = np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)
    output = (patient_id, img_x, img_y, img_z, spacing[1], spacing[2], spacing[0])
    return output




def scan(patient_id, preprocessing_folder):
    filename = '%s.npy' %(patient_id)
    img = np.load(os.path.join(preprocessing_folder,filename))
    img_z = img.shape[0]
    img_x = img.shape[1]
    img_y = img.shape[2]
    output = (patient_id, img_x, img_y, img_z)
    return output

def main():
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
            "--patients_folder",
            metavar="PATIENTS_FOLDER",
            type = str,
            help="Input Folder path that stores patients' folders",
            default= '/workspace/dsbowl2017/data/sample_images'
    )
    parser.add_argument(
            "--label_csv",
            metavar="LABEL_CSV",
            type=str,
            help = "Path of the Label CSV File",
            default = "/workspace/dsbowl2017/data/stage1_labels.csv"
    )
    parser.add_argument(
            "--preprocessing_csv",
            metavar="PREPROCESSING_CSV",
            type=str,
            help= "Path of the preprocessing img CSV File",
            default="/workspace/dsbowl2017/data/preprocessing_img_info.csv"
            )
    parser.add_argument(
            "--preprocessing_folder",
            metavar="PREPROCESSING_FOLDER",
            type=str,
            help="Folder which stores the preprocessed images in .npy",
            default= "/workspace/dsbowl2017/data/preprocessing_images"
    )

    args = parser.parse_args()
    patients = os.listdir(args.patients_folder)
    df_scan_dicom = [scan_dicom(patient_id, args.patients_folder) for patient_id in patients]
    df_scan_dicom = pd.DataFrame.from_records(df_scan_dicom)
    df_scan_dicom.columns = ['id', 'img_x_ori', 'img_y_ori', 'img_z_ori', 'spacing_x_ori', 'spacing_y_ori', 'spacing_z_ori']
    #df_scan = [scan(patient_id,args.preprocessing_folder) for patient_id in patients]
    #df_scan = pd.DataFrame.from_records(df_scan)
    #df_scan.columns = ['id','img_x','img_y','img_z']
    df_scan = pd.read_csv(args.preprocessing_csv)
    df_label = pd.read_csv(args.label_csv)
    df_label = df_label[df_label.id.isin(patients)]
    df_all = pd.merge(df_scan, df_label, on = 'id')
    df_all = pd.merge(df_scan_dicom, df_all, on = 'id')
    name = '%s_info_all.csv' %(os.path.basename(args.patients_folder))
    name = name.replace('images','img')
    filepath = os.path.join(os.path.dirname(args.patients_folder),name)
    df_all.to_csv(filepath)

if __name__=="__main__":
    sys.exit(main())

