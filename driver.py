from numpy import *
import numpy as np
import dicom
import os
import time
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure
from scipy.ndimage import morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.ndimage.filters import gaussian_filter
from scipy.linalg import norm

# constants:
start_time = time.time()
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
MIN_RADIUS = 4
MAX_RADIUS = 16
FILTERS_AMOUNT = 6

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
    #image = image.astype(np.int16)
    image = image.astype(np.float64)

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

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

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
    patient_image, spacing = resample(patient_pixels, patient, [1, 1, 1])
    #patient_image = normalize(patient_image)
    #patient_image = zero_center(patient_image)
    return patient_image

# perfrom region of interest selection
def roi_selection(image):
    segmented_lungs = segment_lung_mask(image, False)
    mask = morphology.binary_fill_holes(
        morphology.binary_dilation(
            morphology.binary_fill_holes(segmented_lungs > 0),
            iterations=4)
    )

    return mask

def get_scales(bottom=MIN_RADIUS, top=MAX_RADIUS,
               amount=FILTERS_AMOUNT):
    radius = (top / bottom) ** (1. / (amount - 1))
    sigmas = [bottom / 4.]
    for i in range(amount - 1):
        sigmas.append(sigmas[0] * (radius ** i + 1))
    return sigmas

def div_of_norm_grad(sigma, patient):
    # Compute gradient
    grad = asarray(gradient(patient))
    # Normalize it with smooth const 1e-3
    grad /= norm(grad, axis=0) + 1e-3
    # Apply gaussian filter aimed to
    grad = [gaussian_filter(deriv, sigma=sigma) for deriv in grad]
    return sum([gradient(el, axis=i)
                for i, el in enumerate(grad)], axis=0)

def maxima_divergence(patient, mask, sigmas):
    divs = list()
    for sigma in sigmas:
        divs.append(div_of_norm_grad(sigma, patient=patient))
    divs = -1 * asarray(divs) * mask
    return divs.max(axis=0)

def feature_extraction(patient, mask):
    sigmas = get_scales()
    mdng = maxima_divergence(patient, mask, sigmas)
    mdng_max = mdng.max()
    mdng_std = mdng.std()
    print("max:\t%f\nstd:\t%f" % (mdng_max, mdng_std))

# main class driver
def driver():
    patient_images = []
    for i in range(len(patients)):
        # stupid mac stuff
        if ".DS_Store" in patients[i]:
            continue

        # pre process and store processed images in list patient_images
        new_time = time.time()
        print("Now pre-processing patient " + str(i))
        proc_image = pre_processing(i)
        patient_images.append(proc_image)
        print("Time to complete pre-processing patient " + str(i) + ": %s seconds.\n" % (time.time() - new_time))

        # segment the lungs and place in mask
        new_time = time.time()
        print("Now performing ROI on patient " + str(i))
        mask = roi_selection(proc_image)
        print("Time to complete ROI selection on patient " + str(i) + ": %s seconds.\n" % (time.time() - new_time))

        new_time = time.time()
        print("Now performing Feature Extraction on patient " + str(i))
        feature_extraction(proc_image, mask)
        print("Time to complete Feature Extraction on patient " + str(i) + ": %s seconds.\n" % (time.time() - new_time))


    # perform feature selection

    # perform classification

driver()
print("--- total running time: %s seconds ---" % (time.time() - start_time))
