import numpy as np
import pandas as pd
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import warnings

from skimage import measure, morphology
from skimage.transform import resize

#import ipyvolume.pylab as p3

#%matplotlib inline
warnings.filterwarnings("ignore")
INPUT_FOLDER = 'sample_images/'
MASK_FOLDER = 'stage1_masks/'
PREPD_FOLDER = 'prepd_stage1/'
IM_SIZE = 128
patients = os.listdir(INPUT_FOLDER)
patients.sort()


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


def get_batches(patients):
    for ix, patient in enumerate(patients):
        scan = load_scan(INPUT_FOLDER + patient)
        slices = get_pixels_hu(scan)
        if ix % 10 == 0:
            print("Processed patient {0} of {1}".format(ix, len(patients)))
        yield scan, slices, patient


def save_array(path, arr):
    np.save(path, arr)
    

def load_array(path):
    return np.load(path)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    
    thickness = [scan[0].SliceThickness]

    if not thickness[0]:
        thickness = [1.0]  # because weird error
        
    spacing = np.array(thickness + scan[0].PixelSpacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image


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
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
        
    for i, axial_slice in enumerate(binary_image):  # one last dilation
        binary_image[i] = morphology.dilation(axial_slice, np.ones([10,10]))
 
    return binary_image


def save_masks(scan, patient):
    masks = segment_lung_mask(scan, True)
    np.save(MASK_FOLDER + "{}.npy".format(patient), masks)
    return masks


def apply_masks(imgs, masks):
    out_images = []
    for i in range(len(imgs)):
        mask = masks[i]
        img = imgs[i]
        img= mask*img  # apply lung mask
        img = resize(img, [IM_SIZE, IM_SIZE]) # resize
        out_images.append(img)
    return np.array(out_images)


def save_preprocessed(patient, scan, masks):
    normalized = normalize(scan)
    centered = zero_center(normalized)
    masked = apply_masks(centered, masks)
    save_array(PREPD_FOLDER + "{}.npy".format(patient), masked)



gen = get_batches(patients)

for scan, slices, patient in gen:
    try:
        resampled = resample(slices, scan)
        masks = save_masks(resampled, patient)
        save_preprocessed(patient, resampled, masks)
    except Exception as e:
        print(patient, e)


demo_gen = get_batches(patients)  # reset generator
im_index = 100                    # choose a random slice from somewhere near the middle of the scan

scan, slices, patient = next(demo_gen)
resampled = resample(slices, scan)

#plt.imshow(resampled[im_index], cmap=plt.cm.gray)


mask = load_array(MASK_FOLDER + '{}.npy'.format(patient))

#plt.imshow(mask[im_index], cmap=plt.cm.gray)

final = load_array(PREPD_FOLDER + '{}.npy'.format(patient))

#plt.imshow(final[im_index], cmap=plt.cm.gray)


#f, plots = plt.subplots(10, 10, sharex='all', sharey='all', figsize=(10, 10))

for i in range(100):
    plots[i // 10, i % 10].axis('off')
    plots[i // 10, i % 10].imshow(final[i+50], cmap=plt.cm.gray)


print("Max pixel value:", np.max(final))
print("Min pixel value:", np.min(final))
print("Mean pixel value:", np.mean(final))
print("Pixel Std. Dev:", np.std(final))



