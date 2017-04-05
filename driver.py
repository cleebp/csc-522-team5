import numpy as np
import pandas as pd
import dicom
import os
import time
import scipy.ndimage
import matplotlib.pyplot as plt

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

# by default patient dicom files arent returned in hounsfield unit (hu) format
# this function takes a dicom slice's pixel array and converts to HU
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

# different dicom scans can have different spacing, this function attempts to resample
# to a standard spacing scale
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

for i in range(len(patients)):
    # stupid mac stuff
    if ".DS_Store" in patients[i]:
        continue
    print("now plotting patient " + str(i))

    patient = load_scan(INPUT_FOLDER + patients[i])
    patient_pixels = get_pixels_hu(patient)
    pix_resampled, spacing = resample(patient_pixels, patient, [1, 1, 1])
    patient_image = normalize(pix_resampled)
    patient_image = zero_center(patient_image)
    filename = "output/patient_" + str(i)

    plt.hist(patient_pixels.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.title("Patient " + str(i) + ": no pre-processing.")
    plt.savefig(filename + "_histo_no-pp.png", bbox_inches='tight')
    plt.close()

    plt.imshow(patient_pixels[80], cmap=plt.cm.gray)
    plt.title("Patient " + str(i) + ": no pre-processing.")
    plt.savefig(filename + "_slice_no-pp.png", bbox_inches='tight')
    plt.close()

    plt.hist(pix_resampled.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.title("Patient " + str(i) + ": resampled pixels.")
    plt.savefig(filename + "_histo-pp1.png", bbox_inches='tight')
    plt.close()

    plt.imshow(pix_resampled[80], cmap=plt.cm.gray)
    plt.title("Patient " + str(i) + ": resampled pixels.")
    plt.savefig(filename + "_slice_pp1.png", bbox_inches='tight')
    plt.close()

    plt.hist(patient_image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.title("Patient " + str(i) + ": resampled pixels, normalized, zero centered.")
    plt.savefig(filename + "_histo_pp2.png", bbox_inches='tight')
    plt.close()

    # Show some slice in the middle
    plt.imshow(patient_image[80], cmap=plt.cm.gray)
    plt.title("Patient " + str(i) + ": resampled pixels, normalized, zero centered.")
    plt.savefig(filename + "_slice_pp2.png", bbox_inches='tight')
    plt.close()

#plot_3d(pix_resampled, 400)
print("--- running time: %s seconds ---" % (time.time() - start_time))