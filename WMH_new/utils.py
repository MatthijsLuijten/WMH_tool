from keras import backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy.spatial
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import parameters
from tqdm import tqdm
import albumentations as A
from sklearn.utils import shuffle


def iou_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)  


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    weights = [0.580184, 3.90522226, 0.71173312, 1.62516944]
    dice=0
    for index in range(numLabels):
        dice += (dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index]) * weights[index])
    return dice


def dice_coef_multilabel_loss(y_true, y_pred):
    return -dice_coef_multilabel(y_true, y_pred)


def getDSC(testImage, resultImage):    
    """Compute the Dice Similarity Coefficient."""
    testArray   = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()
    
    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)
    

def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""
    
    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')
        
    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )
    
    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)    
    
    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)   
        
    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]
        
            
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]
    
    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)    
    
    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
    
    
def getLesionDetection(testImage, resultImage):    
    """Lesion detection metrics, both recall and F1."""
    
    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()    
    ccFilter.SetFullyConnected(True)
    
    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)    
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))
    
    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)
    
    # recall = (number of detected WMH) / (number of true WMH) 
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH
    
    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))
    
    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)
    
    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections
    
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    
    return recall, f1    

    
def getAVD(testImage, resultImage):   
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    
    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)
    result = float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100
    return result


def getAVD2(testImage, resultImage): 
    testImage = sitk.GetArrayFromImage(testImage)  
    resultImage = sitk.GetArrayFromImage(resultImage)  
    result = 100 * (resultImage.sum() - testImage.sum()) / float(testImage.sum())
    return result


def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""
    testImage   = testFilename#sitk.ReadImage(testFilename)
    resultImage = resultFilename#sitk.ReadImage(resultFilename)
    assert testImage.GetSize() == resultImage.GetSize()
    
    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)
    
    # Remove non-WMH from the test and result images, since we don't evaluate on that
    maskedTestImage = sitk.BinaryThreshold(testImage, 0.5,  1.5, 1, 0) # WMH == 1    
    nonWMHImage     = sitk.BinaryThreshold(testImage, 1.5,  2.5, 0, 1) # non-WMH == 2
    maskedResultImage = sitk.Mask(resultImage, nonWMHImage)
    
    # Convert to binary mask
    if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 1000, 1, 0)
    else:
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)
        
    return maskedTestImage, bResultImage

    
def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)


def determine_training(f1, l1, f2, l2, f3, l3, f4, l4, case=''):
    with open('WMH_new/good.txt') as f:
        if case in f.read():
            return
    with open('WMH_new/bad.txt') as f:
        if case in f.read():
            return
    with open('WMH_new/maybe.txt') as f:
        if case in f.read():
            return
    plt.ion()
    fig, ax = plt.subplots(4,2, figsize=(7,15))
    fig.suptitle(case)
    ax[0,0].imshow(np.rot90(f1), cmap='gray')
    ax[0,0].set_title('Original FL')
    ax[0,0].axis('off')

    ax[0,1].imshow(np.rot90(f1), cmap='gray')
    ax[0,1].set_title('FL and LBL')
    ax[0,1].axis('off')

    # ax[0].imshow(np.rot90(label), cmap='hot', alpha=0.5)
    # ax[0].set_title('WMH label')
    # ax[0].axis('off')

    ax[0,1].imshow(np.rot90(l1), cmap='hot', alpha=0.5)
    # ax[1].set_title('WMH label')
    ax[0,1].axis('off')

    ax[1,0].imshow(np.rot90(f2), cmap='gray')
    ax[1,0].set_title('Original FL')
    ax[1,0].axis('off')

    ax[1,1].imshow(np.rot90(f2), cmap='gray')
    ax[1,1].set_title('FL and LBL')
    ax[1,1].axis('off')

    ax[1,1].imshow(np.rot90(l2), cmap='hot', alpha=0.5)
    # ax[1].set_title('WMH label')
    ax[1,1].axis('off')

    ax[2,0].imshow(np.rot90(f3), cmap='gray')
    ax[2,0].set_title('Original FL')
    ax[2,0].axis('off')

    ax[2,1].imshow(np.rot90(f3), cmap='gray')
    ax[2,1].set_title('FL and LBL')
    ax[2,1].axis('off')

    ax[2,1].imshow(np.rot90(l3), cmap='hot', alpha=0.5)
    # ax[1].set_title('WMH label')
    ax[2,1].axis('off')

    ax[3,0].imshow(np.rot90(f4), cmap='gray')
    ax[3,0].set_title('Original FL')
    ax[3,0].axis('off')

    ax[3,1].imshow(np.rot90(f4), cmap='gray')
    ax[3,1].set_title('FL and LBL')
    ax[3,1].axis('off')

    ax[3,1].imshow(np.rot90(l4), cmap='hot', alpha=0.5)
    # ax[1].set_title('WMH label')
    ax[3,1].axis('off')

    plt.show()

    while True:
        user_input = input('\n Use this as training data? (y/n/m): ')
        if user_input.lower() == 'y':
            print(f'✓ {case}')
            with open('WMH_new/good.txt', 'a') as f:
                f.write(case + '\n') 
                f.close()
            break
        elif user_input.lower() == 'n':
            print(f'✗ {case}')
            with open('WMH_new/bad.txt', 'a') as f:
                f.write(case + '\n') 
                f.close()
            break
        elif user_input.lower() == 'm':
            print(f'✓✗ {case}')
            with open('WMH_new/maybe.txt', 'a') as f:
                f.write(case + '\n') 
                f.close()
            break
        else:
            print('Type \'y\' or \'n\' or \'m\'')
            continue

    plt.close()


def get_cmap():
    t1_cmap = LinearSegmentedColormap.from_list('t1_cmap', ['#0D0D0D','#404040','#8C8C8C','#D9D9D9', '#F2F2F2'], N=255)
    fl_cmap = LinearSegmentedColormap.from_list('fl_cmap', ['#0D0D0D','#595959','#8C8C8C','#A6A6A6', '#F2F2F2'], N=255)
    return t1_cmap, fl_cmap


def resize_image(image):
  return tf.image.resize(image, (182,218))


def save_datasets(train_img, train_lbl, test_img, test_lbl):
    # Save datasets
    np.save(parameters.path_train_img, train_img)
    np.save(parameters.path_train_lbl, train_lbl)
    np.save(parameters.path_test_img, test_img)
    np.save(parameters.path_test_lbl, test_lbl)


def save_pm_datasets(train_img, train_lbl_wmh, train_lbl_nawm, train_lbl_gm, test_img, test_lbl_wmh, test_lbl_nawm, test_lbl_gm):
    # Save datasets
    # PM WMH INPUT IMAGES
    np.save(parameters.path_pm_train_img, train_img)
    np.save(parameters.path_pm_test_img, test_img)

    # PM WMH
    np.save(parameters.path_wmh_train_lbl, train_lbl_wmh)
    np.save(parameters.path_wmh_test_lbl, test_lbl_wmh)

    # PM NAWM
    np.save(parameters.path_nawm_train_lbl, train_lbl_nawm)
    np.save(parameters.path_nawm_test_lbl, test_lbl_nawm)

    # PM GM
    np.save(parameters.path_gm_train_lbl, train_lbl_gm)
    np.save(parameters.path_gm_test_lbl, test_lbl_gm)


def normalize_image(image):
    min_val, max_val = np.min(image), np.max(image)
    range = max_val - min_val
    if range == 0:
        return np.array(image)
    else:
        return (image - min_val) / range


def evaluate_metrics(preds, test_lbl):
    # Initialize metrics
    metrics = {}

    # Iterate over predictions and labels
    for p, l in zip(preds, test_lbl):
        # Skip if both prediction and label are empty
        if np.sum(l) == 0:
            continue

        # Compute evaluation metrics
        dsc = round(getDSC(l, p), 3)
        avd = round(getAVD2(l, p), 3)
        recall, f1 = getLesionDetection(l, p)
        recall = round(recall, 3)
        f1 = round(f1, 3)

        # Update metrics
        metrics['DSC'] = metrics.get('DSC', 0) + dsc
        metrics['AVD'] = metrics.get('AVD', 0) + avd
        metrics['Recall'] = metrics.get('Recall', 0) + recall
        metrics['F1'] = metrics.get('F1', 0) + f1

    # Average metrics
    count = len(preds)
    metrics = {k: v / count for k, v in metrics.items()}

    # Print metrics
    print(metrics)


import numpy as np


def augment_data(train_img, train_lbl):
    aug = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, border_mode=0, value=0., p=.5),
            A.GridDistortion(p=0.5, border_mode=0, value=0.),
            # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
            ], p=0.8),
        # A.CLAHE(p=0.8),
        A.GaussNoise(var_limit=(0.1),p=.3),
        A.RandomBrightnessContrast(p=0.8),    
        A.RandomGamma(p=0.8)],
        p=.95)
    
    aug_train_img, aug_train_lbl = [], []
    for i, t in tqdm(enumerate(train_img)):
        for _ in range(50):
            augmented = aug(image=t, mask=train_lbl[i])
            augmented_input = augmented['image']
            augmented_mask = augmented['mask']
            aug_train_img.append(augmented_input)
            aug_train_lbl.append(augmented_mask)

    return shuffle(np.array(aug_train_img), np.array(aug_train_lbl))


def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]