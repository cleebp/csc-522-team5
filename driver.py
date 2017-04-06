import numpy as np
import pandas as pd
import dicom
import os
import time
import scipy.ndimage
import matplotlib.pyplot as plt

from pprint import pprint
from sklearn.cluster import KMeans
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

start_time = time.time()
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

INPUT_FOLDER = 'sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

# load the scans by file path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

# normalize image between -1000 and 400, anything over 400 is unnecesariy bones
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

# take a dicom slice's pixel array and converts it to HU
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

# different dicom scans can have different spacing, this resamples to a standard spacing scale
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

# 3D plot with matplotlib
def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

# pre process image for given patient id, resample pixles, normalize, zero center
def pre_processing(patient_id):
    patient = load_scan(INPUT_FOLDER + patients[patient_id])
    patient_pixels = get_pixels_hu(patient)
    pix_resampled, spacing = resample(patient_pixels, patient, [1, 1, 1])
    patient_image = normalize(pix_resampled)
    patient_image = zero_center(patient_image)
    return patient_image

# perfrom region of interest selection
def roi_selection(image):
    middle = image[100:400, 100:400]
    # perform KMeans looking for two lungs
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    # threshold the image with our cluster centers
    threshold = np.mean(centers)
    thresh_image = np.where(image < threshold, 1.0, 0.0)

    plt.imshow(thresh_image[80], cmap=plt.cm.gray)
    plt.show()

    #return roi_image


def driver():
    patient_images = []
    driver_time = time.time()
    #pre process and store processed images in list patient_images
    for i in range(len(patients)):
        # stupid mac stuff
        if ".DS_Store" in patients[i]:
            continue
        print("Now pre-processing patient " + str(i))
        new_time = time.time()
        proc_image = pre_processing(i)
        patient_images.append(proc_image)
        print("Time to complete pre-processing patient " + str(i) + ": %s seconds.\n" % (time.time() - new_time))

        #placing roi here for now, that way i can debug without waiting 7 minutes, will move below later
        new_time = time.time()
        print("Now performing ROI on patient " + str(i))
        roi_selection(proc_image)
        print("Time to complete ROI selection on patient " + str(i) + ": %s seconds.\n" % (time.time() - new_time))

    print("Pre processed: " + str(len(patient_images)) + " patients in %s seconds." % (time.time() - driver_time))
    # perform roi

    # perform feature selection

    # perform classification

driver()
print("--- total running time: %s seconds ---" % (time.time() - start_time))