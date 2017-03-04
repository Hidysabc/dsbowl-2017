#!/usr/bin/env python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import argparse
import logging
import sys
from preprocessing import load_scan, get_pixels_hu, resample, resample_new, normalize, zero_center, cropping, shift, harmonize

FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOG_MAP = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warn": logging.WARNING
        }

logging.basicConfig(format=FORMAT)
logger = logging.getLogger('preprocessing_local')
logger.setLevel(LOG_MAP["debug"])


def main():
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        type=str,
        help = "Input folder, the folder which stores a collection of patients'CT scanned folders"
    )
    parser.add_argument(
        "--output",
        metavar="OUTPUT",
        type=str,
        help = "Output folder, the folder which stores a collection of preprocessing images in *.npy format",
        default= "/workspace/dsbowl2017/data/preprocessing_images_all/"
    )
    parser.add_argument(
        "--logging",
        metavar="LOG_LEVEL",
        type =str,
        help = "Log level",
        default = "debug"
    )
    args = parser.parse_args()

    patients = os.listdir(args.input)
    patients.sort()
    patients_scans = [load_scan(os.path.join(args.input,id)) for id in patients]
    patients_img = [get_pixels_hu(patient_scans) for patient_scans in patients_scans]
    
    #patients_pixels_resampled, patients_spacing = [resample(patient_img, patient_scans, [2,1,1]) for patient_img,  patient_scans in zip(patients_img, patients_scans)]
    directory = os.path.join(args.output)
    if not os.path.exists(directory):
        os.makedirs(directory)
        patients_pixels_resampled = []
        patients_spacing = []
        output = []
        for i in np.arange(len(patients)):
            #patient_pixels_resampled, patient_spacing = resample(patients_img[i], patients_scans[i], [2,1,1])
            patient_pixels_resampled, patient_spacing = resample_new(patients_img[i], patients_scans[i], [128,256,256])
            patients_pixels_resampled.append(patient_pixels_resampled)
            patients_spacing.append(patient_spacing)
            patient_pixels_normalized = normalize(patient_pixels_resampled)
            patient_pixels_zero_centered = zero_center(patient_pixels_normalized)
            img_z_size = patient_pixels_resampled.shape[0]
            img_x_size = patient_pixels_resampled.shape[1]
            img_y_size = patient_pixels_resampled.shape[2]
            output.append((patients[i],img_x_size, img_y_size, img_z_size, patient_spacing[1], patient_spacing[2], patient_spacing[0]))
        
            print((output))
            logger.debug("Shape before resampling\t{}".format(patients_img[i].shape))
            logger.debug("Shape after resampling\t{}".format(patient_pixels_resampled.shape))
            np.save(os.path.join(directory,"%s.npy" %(patients[i])), patient_pixels_zero_centered)
        df_output = pd.DataFrame.from_records(output)
        df_output.columns = ['id', 'img_x_size', 'img_y_size', 'img_z_size','spacing_x', 'spacing_y', 'spacing_z']
        df_output.to_csv('/workspace/dsbowl2017/data/preprocessing_img_info.csv',index=False)
    else:
        logger.debug("Pre-processed folder: {}  already exists!".format(directory))

if __name__ == "__main__":
    sys.exit(main())

