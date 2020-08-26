import tensorflow as tf

def dice_loss_v2(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
    return 1 - numerator / denominator

def wbce(y_true, y_pred, weight1 = 10.0, weight0 = 1.0):
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logloss = -(y_true * tf.keras.backend.log(y_pred) * weight1 + (1 - y_true) * tf.keras.backend.log(1 - y_pred) * weight0)
    return tf.keras.backend.mean(logloss, axis=-1)

def CreateUNet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, num_classes):
    # Input
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) # 512 x 640 x 3
    s = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs) # 512 x 640 x 3
    
    # Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s) # 512 x 640 x 16
    c1 = tf.keras.layers.Dropout(0.1)(c1) # 512 x 640 x 16
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1) # 512 x 640 x 16
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) # 256 x 320 x 16
    
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1) # 256 x 320 x 32
    c2 = tf.keras.layers.Dropout(0.1)(c2) # 256 x 320 x 32
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2) # 256 x 320 x 32
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) # 128 x 160 x 32
    
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2) # 128 x 160 x 64
    c3 = tf.keras.layers.Dropout(0.2)(c3) # 128 x 160 x 64
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3) # 128 x 160 x 64
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3) # 64 x 80 x 64
    
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3) # 64 x 80 x 128
    c4 = tf.keras.layers.Dropout(0.2)(c4) # 64 x 80 x 128
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4) # 64 x 80 x 128
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4) # 32 x 40 x 128
    
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4) # 32 x 40 x 256
    c5 = tf.keras.layers.Dropout(0.3)(c5) # 32 x 40 x 256
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5) # 32 x 40 x 256
    
    # Classification branch
    cl1 = tf.keras.layers.Flatten()(c5)
    cl1 = tf.keras.layers.Dense(128)(cl1)
    cl1 = tf.keras.layers.Activation('relu')(cl1)
    cl1 = tf.keras.layers.BatchNormalization()(cl1)
    cl1 = tf.keras.layers.Dropout(0.5)(cl1)
    cl1 = tf.keras.layers.Dense(num_classes)(cl1)
    
    classificaion_output = tf.keras.layers.Activation('softmax', name = "class_output")(cl1)
    
    # Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) # 64 x 80 x 128
    u6 = tf.keras.layers.concatenate([u6, c4]) # 64 x 80 x 256
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6) # 64 x 80 x 128
    c6 = tf.keras.layers.Dropout(0.2)(c6) # 64 x 80 x 128
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6) # 64 x 80 x 128
    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6) # 128 x 160 x 64
    u7 = tf.keras.layers.concatenate([u7, c3]) # 128 x 160 x 128
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7) # 128 x 160 x 64
    c7 = tf.keras.layers.Dropout(0.2)(c7) # 128 x 160 x 64
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7) # 128 x 160 x 64
    
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) # 256 x 320 x 32
    u8 = tf.keras.layers.concatenate([u8, c2]) # 256 x 320 x 64
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8) # 256 x 320 x 32
    c8 = tf.keras.layers.Dropout(0.1)(c8) # 256 x 320 x 32
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8) # 256 x 320 x 32
    
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8) # 512 x 640 x 16
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3) # 512 x 640 x 32
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9) # 512 x 640 x 16
    c9 = tf.keras.layers.Dropout(0.1)(c9) # 512 x 640 x 16
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9) # 512 x 640 x 16
    
    segmentaion_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name = "segm_output")(c9) # 512 x 640 x 1
    
    model = tf.keras.Model(inputs = [inputs], outputs = [classificaion_output, segmentaion_output])
    
    return model