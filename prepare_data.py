import json
import os
import cv2
import numpy as np
from tqdm import tqdm

seed = 14
np.random.seed = seed

ORIG_TRAIN_IMG_PATH = 'dataset/train/img_orig'
TRAIN_IMG_PATH = 'dataset/train/img'
TEST_IMG_PATH = 'dataset/test/img'

ORIG_TRAIN_MASK_PATH = 'dataset/train/mask_orig'
TRAIN_MASK_PATH = 'dataset/train/mask'

TRAIN_JSON_PATH = 'dataset/train/json'

JSON_FILE_FILTER = '.json'
PNG_FILE_FILTER = '.png'

LABELS = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae', 'staphylococcus_aureus', 'moraxella_catarrhalis', 'c_kefir', 'ent_cloacae']


def elastic_image_transform(image, seed = 0, alpha = 1800, sigma = 100, alpha_affine = 100):
    random_state = np.random.RandomState(seed)

    shape_size = image.shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size = pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode = cv2.BORDER_REFLECT_101)

    blur_size = int(4 * sigma) | 1
    rand_x = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1).astype(np.float32), ksize = (blur_size, blur_size), sigmaX = sigma) * alpha
    rand_y = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1).astype(np.float32), ksize = (blur_size, blur_size), sigmaX = sigma) * alpha

    grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    grid_x = (grid_x + rand_x).astype(np.float32)
    grid_y = (grid_y + rand_y).astype(np.float32)

    distorted_img = cv2.remap(image, grid_x, grid_y, borderMode = cv2.BORDER_REFLECT_101, interpolation = cv2.INTER_LINEAR)

    return distorted_img

# Create mask for one image
def create_mask(json_filename):
    with open(os.path.join(TRAIN_JSON_PATH, json_filename), 'r') as f:
        layout = json.load(f)

    h, w = layout['imageHeight'], layout['imageWidth']
    true_mask = np.zeros((w, h), np.uint8)

    for shape in layout['shapes']:
        polygon = np.array([point[::-1] for point in shape['points']])
        cv2.fillPoly(true_mask, [polygon], 255)

    mask_filename = json_filename[:-len(JSON_FILE_FILTER)] + "_mask.png"

    cv2.imwrite(os.path.join(ORIG_TRAIN_MASK_PATH, mask_filename), np.transpose(true_mask))

# Create masks for all original images
def create_masks():
    json_files = os.listdir(TRAIN_JSON_PATH)
    for json_filename in json_files:
        create_mask(json_filename)

# Flip images and masks (train data x 4)
def augment_train_data_flip_transform():
    images = os.listdir(ORIG_TRAIN_IMG_PATH)
    rand_seed = 0
    
    for image_name in tqdm(images, total = len(images)):
        root_name = image_name[:-len(PNG_FILE_FILTER)]
        mask_name = root_name + "_mask.png"
        
        # Read original image and mask
        img_0 = cv2.imread(os.path.join(ORIG_TRAIN_IMG_PATH, image_name), cv2.IMREAD_COLOR)
        mask_0 = cv2.imread(os.path.join(ORIG_TRAIN_MASK_PATH, mask_name), cv2.IMREAD_GRAYSCALE)
        
        # Write original image and mask
        cv2.imwrite(os.path.join(TRAIN_IMG_PATH, root_name + '_0.png'), img_0)
        cv2.imwrite(os.path.join(TRAIN_MASK_PATH, root_name + '_0_mask.png'), mask_0)
        
        # Write vertically flipped image and mask
        img_1 = cv2.flip(img_0, 0)
        mask_1 = cv2.flip(mask_0, 0)
        cv2.imwrite(os.path.join(TRAIN_IMG_PATH, root_name + '_1.png'), img_1)
        cv2.imwrite(os.path.join(TRAIN_MASK_PATH, root_name + '_1_mask.png'), mask_1)
        
        # Write horizontally flipped image and mask
        img_2 = cv2.flip(img_0, 1)
        mask_2 = cv2.flip(mask_0, 1)
        cv2.imwrite(os.path.join(TRAIN_IMG_PATH, root_name + '_2.png'), img_2)
        cv2.imwrite(os.path.join(TRAIN_MASK_PATH, root_name + '_2_mask.png'), mask_2)
        
        # Write vertically and horizontally flipped image and mask
        img_3 = cv2.flip(img_0, -1)
        mask_3 = cv2.flip(mask_0, -1)
        cv2.imwrite(os.path.join(TRAIN_IMG_PATH, root_name + '_3.png'), img_3)
        cv2.imwrite(os.path.join(TRAIN_MASK_PATH, root_name + '_3_mask.png'), mask_3)
        
        # Elastic deformations (6 different)
        imgs = [img_0, img_1, img_2, img_3]
        masks = [mask_0, mask_1, mask_2, mask_3]
        for transform_ind in range(4, 10):
            img_ind = np.random.randint(0, 3)
            img_to_transform = imgs[img_ind]
            mask_to_transform = masks[img_ind]
            img_transformed = elastic_image_transform(img_to_transform, seed = rand_seed)
            mask_transformed = elastic_image_transform(mask_to_transform, seed = rand_seed)
            cv2.imwrite(os.path.join(TRAIN_IMG_PATH, root_name + '_' + str(transform_ind) + '.png'), img_transformed)
            cv2.imwrite(os.path.join(TRAIN_MASK_PATH, root_name + '_' + str(transform_ind) + '_mask.png'), mask_transformed)
            rand_seed += 1
            
def augment_train_data_crop():
    return None

# Create data for training
def prepare_train_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, shuffle_data = False):
    images = os.listdir(TRAIN_IMG_PATH)
    
    X_train = np.zeros((len(images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    Y_class_train = np.zeros((len(images), len(LABELS)), dtype = np.uint8)
    Y_segm_train = np.zeros((len(images), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.uint8)
    
    # Shuffle ind
    ind_map = list(range(len(images)))
    np.random.shuffle(ind_map)
    
    for ind, image_name in tqdm(enumerate(images), total = len(images)):
        if (shuffle_data):
            n = ind_map[ind]
        else:
            n = ind
        
        # Fill X_train
        img = cv2.imread(os.path.join(TRAIN_IMG_PATH, image_name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X_train[n] = img
        
        # Fill Y_segm_train
        mask_name = image_name[:-len(PNG_FILE_FILTER)] + "_mask.png"
        mask = cv2.imread(os.path.join(TRAIN_MASK_PATH, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = np.expand_dims(mask, axis = -1)
        mask = (mask > 0).astype(np.uint8)
        Y_segm_train[n] = mask
        
        # Fill Y_class_train
        json_name = image_name[0:3] + JSON_FILE_FILTER
        with open(os.path.join(TRAIN_JSON_PATH, json_name), 'r') as f:
            layout = json.load(f)
        class_label = layout['shapes'][0]['label']
        class_index = LABELS.index(class_label)
        Y_class_train[n, class_index] = 1
        
    train_data = dict();
    train_data['X_train'] = X_train
    train_data['Y_class_train'] = Y_class_train
    train_data['Y_segm_train'] = Y_segm_train
    return train_data

# Create data for testing
def prepare_test_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    images = os.listdir(TEST_IMG_PATH)
    
    X_test = np.zeros((len(images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    
    for n, image_name in tqdm(enumerate(images), total = len(images)):
        # Fill X_train
        img = cv2.imread(os.path.join(TEST_IMG_PATH, image_name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X_test[n] = img
        
    test_data = dict();
    test_data['X_test'] = X_test
    return test_data 
 
######################################################################

create_masks()
augment_train_data()

prepare_train_data(256, 320, 3)
prepare_test_data(256, 320, 3)