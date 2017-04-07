#!/usr/bin/env python
"""
Performing preprocessing of one patients scan images

Functions were taken from tutorial on Kaggle:
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
by Guido Zuidhof
"""

from __future__ import print_function
from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import argparse
import logging
import sys
import boto3
import shutil

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
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path) if '.dcm'in s]
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

def resample_new(image, scan, target_size=[128,256,256]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    real_resize_factor = np.array(target_size) / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image


def cropping(image, x_size, y_size):
    x_mean = image.shape[1]//2
    y_mean = image.shape[2]//2
    crop = image[:,x_mean-x_size/2:x_mean+x_size/2,y_mean-y_size/2:y_mean+y_size/2]
    return crop


def shift(zoom_img, target_size, max_index):
    image_x_size = zoom_img.shape[1]
    image_y_size = zoom_img.shape[2]
    image_z_size = zoom_img.shape[0]
    if max_index == 0:
        shift_0 = 0
        shift_1 = int((target_size-image_x_size)/2)
        shift_2 = int((target_size-image_y_size)/2)
    elif max_index ==1:
        shift_0 = int((target_size-image_z_size)/2)
        shift_1 = 0
        shift_2 = int((target_size-image_y_size)/2)
    else:
        shift_0 = int((target_size-image_z_size)/2)
        shift_1 = int((target_size-image_x_size)/2)
        shift_2 = 0
    shift = (shift_0, shift_1, shift_2)

    return shift


def harmonize(img, target_size):
    max_size = np.max(img.shape)
    max_index = np.argmax(img.shape)
    scale_ratio = target_size/max_size
    zoom_image= scipy.ndimage.interpolation.zoom(img, scale_ratio, mode='nearest')
    image_x_size = zoom_image.shape[1]
    image_y_size = zoom_image.shape[2]
    image_z_size = zoom_image.shape[0]
    zoom_image_mode = stats.mode(zoom_image[int(zoom_image.shape[0]/2*scale_ratio)].flatten())[0]
    new_img = np.ndarray([target_size,target_size,target_size], dtype=np.float32)
    new_img = new_img+zoom_image_mode
    #print(new_img.shape)
    #print(zoom_image.shape)
    shift_all =shift(zoom_image, target_size, max_index)
    new_img[shift_all[0]:shift_all[0]+image_z_size:,shift_all[1]:shift_all[1]+image_x_size,shift_all[2]:shift_all[2]+image_y_size]=zoom_image
    return new_img


def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        print('successfully remove file from {}!'.format(path))


def delete_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print('successfully remove directory from {}!'.format(path))




def main():
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        metavar="INPUT",
        type=str,
        help = "Input folder, the folder which stores a collection of patients' CT scanned folders",
        default="sample_images"
    )
    parser.add_argument(
        "--count",
        metavar="COUNT",
        type=int,
        help= "Number of patient folders to be downloaded and preprocessed",
        default=10
    )
    parser.add_argument(
        "--s3bucket",
        metavar = "S3BUCKET",
        type = str,
        help = "AWS S3 bucket name where the data resides",
        default = "dsbowl2017-sample-images"
    )
    parser.add_argument(
        "--logging",
        metavar="LOG_LEVEL",
        type =str,
        help = "Log level",
        default = "debug"
    )
    parser.add_argument(
        "--output",
        metavar="OUTPUT",
        type=str,
        help = "Output folder, the folder which stores a collection of preprocessing images in *.npy format",
        default= "preprocessing_images"
    )
    parser.add_argument(
        "--labels_csv",
        metavar="LABELS_CSV",
        type=str,
        help = "stage1 csv files which stores the info of images' ids and cancer",
        default= "stage1_labels.csv"
    )

    args = parser.parse_args()
    input_dir = '/tmp/'
    #patient_id = args.input
    filetype = '.npy'

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(args.s3bucket)
    all_keys = [obj.key for obj in bucket.objects.all()]
    bucket_all_keys = all_keys
    all_keys = [i for i in all_keys if args.input in i]
    patient_id_keys = [filename.split('/') for filename in all_keys if len(filename.split('/'))>2]
    patient_id_keys = [i[1] for i in patient_id_keys]
    patient_id = np.unique(patient_id_keys)
    patient_id.sort()

    s3_client = boto3.client('s3')
    patients_pixels_resampled = []
    patients_spacing = []
    output = []

    #while(len(patient_id))>0:
    for x in np.arange(int(len(patient_id)/args.count+0.5)):
        tmp_patient_id = patient_id[x*args.count:(x+1)*args.count]
        for i in np.arange(len(tmp_patient_id)):
            patient_imgs_keys = [j for j in all_keys if tmp_patient_id[i] in j]
            pathname = os.path.join(input_dir, os.path.dirname(patient_imgs_keys[0]).split('/')[1])
            if os.path.join(args.output,tmp_patient_id[i]+filetype) in bucket_all_keys:
                logger.debug("Pre-processed image: {}.npy  already exists in S3!".format(tmp_patient_id[i]))

            else:
                if not os.path.exists(pathname):
                    os.mkdir(pathname)
                [s3_client.download_file(args.s3bucket, key, os.path.join(pathname,os.path.basename(key))) for key in patient_imgs_keys]
                logger.debug("Successfully downloaded images from S3 patient_id:{}!".format(tmp_patient_id[i]))
            '''
            objs = [bucket.objects.filter(Prefix=key) for key in patient_imgs_keys]
            for obj in objs:
                s3.Object(bucket.name, obj.key).delete()
            '''
        patients = os.listdir(input_dir)
        if len(patients)>0:
            patients.sort()
            patients_scans = [load_scan(os.path.join(input_dir,id)) for id in patients]
            patients_img = [get_pixels_hu(patient_scans) for patient_scans in patients_scans]

        #patients_pixels_resampled = []
        #patients_spacing = []
            for i in np.arange(len(patients)):
                filename = "%s%s" %(patients[i],filetype)
                newPath = os.path.join(input_dir,filename)
                if not os.path.exists(newPath):
                    patient_pixels_resampled, patient_spacing = resample_new(patients_img[i], patients_scans[i], [128,256,256])
                    patients_pixels_resampled.append(patient_pixels_resampled)
                    patients_spacing.append(patient_spacing)
                    patient_pixels_normalized = normalize(patient_pixels_resampled)
                    patient_pixels_zero_centered = zero_center(patient_pixels_normalized)

                    img_z_size = patient_pixels_resampled.shape[0]
                    img_x_size = patient_pixels_resampled.shape[1]
                    img_y_size = patient_pixels_resampled.shape[2]

                    output.append((patients[i],img_x_size, img_y_size, img_z_size, patient_spacing[1], patient_spacing[2], patient_spacing[0]))

                    #print((output))
                    logger.debug("Shape before resampling\t{}".format(patients_img[i].shape))
                    logger.debug("Shape after resampling\t{}".format(patient_pixels_resampled.shape))
                    np.save(newPath, patient_pixels_zero_centered)
                    s3_client.upload_file(newPath,args.s3bucket,os.path.join(args.output,os.path.basename(newPath)))

                    shutil.rmtree(os.path.join(input_dir,patients[i]))
                    logger.debug("Successfully uploaded image: {} to S3".format(newPath))

                    os.remove(newPath)
                else:
                    logger.debug("Pre-processed image: {}  already exists!".format(newPath))


        #patient_id = np.delete(patient_id, np.arange(args.count))
    if len(output)>0:
        df_output = pd.DataFrame.from_records(output)
        df_output.columns = ['id', 'img_x_size', 'img_y_size', 'img_z_size','spacing_x', 'spacing_y', 'spacing_z']
        df_output.to_csv(os.path.join(input_dir,'preprocessing_img_info_%d.csv' %(args.count)),index=False)
        s3_client.upload_file(os.path.join(input_dir, 'preprocessing_img_info_%d.csv' %(args.count)), args.s3bucket, 'preprocessing_img_info_%d.csv' %(args.count))
        logger.debug("Successfully uploaded csv: {} to S3".format('preprocessing_img_info_%d.csv' %(args.count)))
        df_output.set_index("id", drop=True, inplace=True)

        labels_csv = args.labels_csv
        s3_client.download_file(args.s3bucket, labels_csv, os.path.join(input_dir, labels_csv))
        labels_csv_path = os.path.join(input_dir, labels_csv)
        #df_scan = pd.read_csv(args.preprocessing_csv)
        df_label = pd.read_csv(labels_csv_path)
        df_label = df_label[df_label.id.isin(df_output.index)]
        df_all = df_label.join(df_output, on = 'id')
        #df_all = pd.merge(df_scan_dicom, df_all, on = 'id')
        name = 'sample_img_info_all.csv'
        filepath = os.path.join(input_dir,name)
        df_all.to_csv(filepath, index=False)
        s3_client.upload_file(filepath, args.s3bucket, os.path.basename(filepath))
        logger.debug("Successfully uploaded csv: {} to S3".format(os.path.basename(filepath)))
    else:
        logger.debug("Already preprocess all the images!!!!")

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



