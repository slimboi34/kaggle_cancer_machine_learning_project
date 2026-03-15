import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_dual_path_model():
    input_img = Input(shape=(96, 96, 3), name='full_image_input')
    
    # --- Base/Context Stream (receives full 96x96 image) ---
    x_full = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x_full = BatchNormalization()(x_full)
    x_full = MaxPooling2D((2, 2))(x_full) # 48x48
    
    x_full = Conv2D(64, (3, 3), activation='relu', padding='same')(x_full)
    x_full = MaxPooling2D((2, 2))(x_full) # 24x24
    
    x_full = Conv2D(128, (3, 3), activation='relu', padding='same', name='full_path_last_conv')(x_full)
    x_full = MaxPooling2D((2, 2))(x_full) # 12x12
    x_full = Flatten()(x_full)
    
    # --- Center Stream (Extracts center 32x32 image) ---
    # Crop the center 32x32 pixels: [start_y:end_y, start_x:end_x]
    # For a 96x96 image, the center 32x32 is from indices 32 to 64.
    x_center = tf.keras.layers.Cropping2D(cropping=((32, 32), (32, 32)))(input_img)
    
    x_center_path = Conv2D(32, (3, 3), activation='relu', padding='same')(x_center)
    x_center_path = BatchNormalization()(x_center_path)
    x_center_path = MaxPooling2D((2, 2))(x_center_path) # 16x16
    
    x_center_path = Conv2D(64, (3, 3), activation='relu', padding='same')(x_center_path)
    x_center_path = MaxPooling2D((2, 2))(x_center_path) # 8x8
    
    x_center_path = Conv2D(128, (3, 3), activation='relu', padding='same', name='center_path_last_conv')(x_center_path)
    x_center_path = MaxPooling2D((2, 2))(x_center_path) # 4x4
    x_center_path = Flatten()(x_center_path)
    
    # --- Concatenation & Output ---
    merged = Concatenate()([x_full, x_center_path])
    
    z = Dense(256, activation='relu')(merged)
    z = Dropout(0.5)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.3)(z)
    
    output = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=input_img, outputs=output, name='CenterFocusDualPathCNN')
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model
