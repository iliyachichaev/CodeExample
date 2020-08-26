import tensorflow as tf
import numpy as np
import cv2
import os
from prepare_data import prepare_train_data, prepare_test_data
from utils import wbce

LABELS = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae', 'staphylococcus_aureus', 'moraxella_catarrhalis', 'c_kefir', 'ent_cloacae']

TRAIN_SEGM_RESULT_PATH = 'dataset/train/result_segm'
TEST_SEGM_RESULT_PATH = 'dataset/test/result_segm'
TRAIN_CLASS_RESULT_PATH = 'dataset/train/result_class'
TEST_CLASS_RESULT_PATH = 'dataset/test/result_class'

IMG_HEIGHT = 512
IMG_WIDTH = 640
IMG_CHANNELS = 3

def save_results_to_files(class_predictions, segm_predictions, mode):
    # Define folders to save
    if (mode == 'train'):
        segm_path = TRAIN_SEGM_RESULT_PATH
        class_path = TRAIN_CLASS_RESULT_PATH
    else:
        segm_path = TEST_SEGM_RESULT_PATH
        class_path = TEST_CLASS_RESULT_PATH
        
    # Threshold segmentation results
    segm_predictions_t = (segm_predictions > 0.75).astype(np.uint8)
    
    for i in range(segm_predictions_t.shape[0]):
        # Construct file name root
        if (mode == 'train'):
            augm_num = 10
            prefix = (i // augm_num) + 1
            postfix = i % augm_num
            filename_root = str(prefix).zfill(3) + '_' + str(postfix)
        else:
            filename_root = str(i + 1).zfill(3)
        
        # Save result mask
        segm_filename = filename_root + "_result.png"
        result = segm_predictions_t[i] * 255
        if (mode == 'train'):
            cv2.imwrite('dataset_its/train/temp/result_small/' + segm_filename, result)
        result = cv2.resize(result, (640, 512), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(segm_path, segm_filename), result)
        
        # Save result class name
        result_class_index =  np.argmax(class_predictions[i])
        result_class_name = LABELS[result_class_index]
        class_filename = filename_root + "_class.txt"
        with open(os.path.join(class_path, class_filename), 'w') as f:
            f.write(result_class_name)


# Load model
model = tf.keras.models.load_model('bacteria_model.h5', compile = False)
losses = {"class_output": "categorical_crossentropy", "segm_output": wbce}
lossWeights = {"class_output": 1.0, "segm_output": 1.0}
model.compile(optimizer = 'adam', loss = losses, loss_weights = lossWeights, metrics=['accuracy'])

# Get training data
train_data = prepare_train_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, False)
X_train = train_data['X_train']

# Get testing data
test_data = prepare_test_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
X_test = test_data['X_test']

# Get results
[class_predictions_train, segm_predictions_train] = model.predict(X_train, verbose = 1)
[class_predictions_test, segm_predictions_test] = model.predict(X_test, verbose = 1)

save_results_to_files(class_predictions_train, segm_predictions_train, 'train')
save_results_to_files(class_predictions_test, segm_predictions_test, 'test')
