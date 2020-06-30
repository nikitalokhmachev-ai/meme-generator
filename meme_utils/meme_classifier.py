import os
import shutil

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm

class MemeClassifier:
    
    img_height=150
    img_width=150
    
    def __init__(self):
        
        self.model = None
        self.idx_to_classes = {0: 'boxes', 1: 'no_boxes'}
                        
    def __model_init(self):
        
        self.model = Sequential([
            Conv2D(16, 5, padding='same', activation='relu', input_shape=(self.img_height, self.img_width ,3)),
            MaxPooling2D(),
            Conv2D(32, 5, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 5, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
    
    def train(self, train_dir, validation_dir, model_path, epochs=10, batch_size=128):
        
        assert 'boxes' in os.listdir(train_dir)
        assert 'no_boxes' in os.listdir(train_dir)
        assert 'boxes' in os.listdir(validation_dir)
        assert 'no_boxes' in os.listdir(validation_dir)
        
        train_boxes_dir = os.path.join(train_dir, 'boxes')  
        train_no_boxes_dir = os.path.join(train_dir, 'no_boxes')  
        validation_boxes_dir = os.path.join(validation_dir, 'boxes')
        validation_no_boxes_dir = os.path.join(validation_dir, 'no_boxes')  
        
        num_boxes_tr = len(os.listdir(train_boxes_dir))
        num_no_boxes_tr = len(os.listdir(train_no_boxes_dir))
        num_boxes_val = len(os.listdir(validation_boxes_dir))
        num_no_boxes_val = len(os.listdir(validation_no_boxes_dir))
        
        total_train = num_boxes_tr + num_no_boxes_tr
        total_val = num_boxes_val + num_no_boxes_val
        
        train_image_generator = ImageDataGenerator(rescale=1./255)
        validation_image_generator = ImageDataGenerator(rescale=1./255)
        
        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(self.img_height, self.img_width),
                                                           class_mode='binary')
        
        val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(self.img_height, self.img_width),
                                                              class_mode='binary')
        
        self.__model_init()
        
        self.model.fit_generator(
            train_data_gen,
            steps_per_epoch=total_train // batch_size,
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=total_val // batch_size
        )
            
        self.model.save_weights(os.path.join(model_path, "model.h5"))
        
    def predict(self, input_images_path):
        
        class_folder = os.path.basename(os.path.normpath(input_images_path))
        path_to_class_folder = os.path.split(input_images_path)[0]
        
        test_image_generator = ImageDataGenerator(rescale=1./255)
        
        test_data_gen = test_image_generator.flow_from_directory(directory=path_to_class_folder,
                                                                 classes=[class_folder],
                                                                 class_mode=None,
                                                                 shuffle=False,
                                                                 target_size=(self.img_height, self.img_width))
        if self.model is not None:
            try:
                preds = self.model.predict_generator(test_data_gen)
            except Exception as e:
                print(f'Error: {e}')
        else:
            raise Exception("The model was not initialized.")
            
        pred_classes_func = np.vectorize(lambda x: 1 if x > 0.5 else 0)
        preds = pred_classes_func(preds)
        
        preds_classes = np.vectorize(self.idx_to_classes.get)(preds)
        preds_classes = [el[0] for el in preds_classes]
        
        filenames_to_classes = list(zip(test_data_gen.filenames, preds_classes))
        
        images_classes_df = pd.DataFrame(filenames_to_classes, columns=['Image','Class'])
        
        return images_classes_df
        
    def predict_and_filter(self, input_images_path, output_images_path):
        
        print('Predicting and filtering...')
        
        images_classes_df = self.predict(input_images_path)
        
        only_boxes = images_classes_df[images_classes_df.Class=='boxes']['Image'].to_numpy()
        only_boxes = [os.path.basename(el) for el in only_boxes]
        
        for i in tqdm(range(0,len(only_boxes))):
            
            source = os.path.join(input_images_path, only_boxes[i])
            destination = os.path.join(output_images_path, only_boxes[i])
                
            shutil.copy(source, destination)
        
    def load_model(self, model_path):
        
        self.__model_init()
        
        self.model.load_weights(model_path)
