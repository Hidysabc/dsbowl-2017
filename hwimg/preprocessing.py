#!/usr/bin/env python
"""
Performing preprocessing of one patients scan images

Functions were taken from tutorial on Kaggle:
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
by Guido Zuidhof
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import argparse
import logging
import sys


FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOG_MAP = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warn": logging.WARNING
        }

logging.basicConfig(format=FORMAT)
logger = logging.getLogger('preprocessing')

logger.setLevel(LOG_MAP["debug"])


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
                                                                         
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def main():
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        type=str,
        help = "Folder storing patient's scanned images"
    )
    parser.add_argument(
        "--logging",
        metavar="LOG_LEVEL",
        type =str,
        help = "Log level",
        default = "debug"
    )
    args = parser.parse_args()

    input_dir = args.input
    filetype = '.npy'
    patient_id = os.path.basename(input_dir)
    newPath = "%s.npy" %(patient_id)
    if not os.path.exists(newPath):
        patient_scan = load_scan(input_dir)
        patient_pixels = get_pixels_hu(patient_scan)

        patient_pixels_resampled, patient_spacing = resample(patient_pixels, patient_scan, [1,1,1])
        patient_pixels_normalized = normalize(patient_pixels_resampled)
        patient_pixels_zero_centered = zero_center(patient_pixels_normalized)
        logger.debug("Shape before resampling\t{}".format(patient_pixels.shape))
        logger.debug("Shape after resampling\t{}".format(patient_pixels_resampled.shape))
        
        np.save(newPath,patient_pixels_zero_centered)
        logger.debug("new img: {} adds to processed collection!".format(newPath))
    else:
        logger.debug("processed imgs: {}  already exists!".format(newPath))

if __name__ == "__main__":
    sys.exit(main())



'''
patient_id = '00cba091fa4ad62cc3200a657aeb957e'
single_patient_folder = INPUT_FOLDER+patient_id
single_patient_dir = os.listdir(single_patient_folder)
single_patient = load_scan(single_patient_folder)
single_patient_pixels = get_pixels_hu(single_patient)
pix_resampled, spacing = resample(single_patient_pixels, single_patient, [1,1,1])
normalized = normalize(pix_resampled)
zero_centered = zero_center(normalized)

print("Shape before resampling\t{}".format(single_patient_pixels.shape))
print("Shape after resampling\t{}".format(pix_resampled.shape))
'''
#directory = os.path.join(output_path)
#if not os.path.exists(directory):
#    os.makedirs(directory)


#np.save(os.path.join(directory,"%s.npy" %(patient_id)), zero_centered)



