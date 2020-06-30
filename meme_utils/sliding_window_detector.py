import os
import logging

import numpy as np
from PIL import Image, ImageChops

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.utils import shuffle
from skimage import morphology
from tqdm import tqdm



config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


class ImageSeparator:
    
    img_height=32
    img_width=32
    
    def __init__(self):
        
        self.model = None
        self.idx_to_classes = {'borders': 0, 'images': 1}
                        
    def __model_init(self, img_height, img_width):
        
        self.model = Sequential([
            Conv2D(16, 5, padding='valid', activation='relu', input_shape=(self.img_height, self.img_width ,3)),
            Conv2D(16, 2, padding='valid', activation='relu'),
            MaxPooling2D(),
            Conv2D(16, 3, padding='valid', activation='relu'),
            MaxPooling2D(),
            Conv2D(400, 5, padding='valid', activation='relu'),
            Conv2D(400, 1, padding='same', activation='relu'), 
            Conv2D(2, 1, padding='same', activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
        
    def convert_targets(self, target_array):
        
        image_label = np.zeros((1,1,2))
        border_label = np.zeros((1,1,2))
        image_label[0,0,1] = 1
        border_label[0,0,0] = 1
        
        target_array_size = target_array.shape[0]
        
        new_target_array = np.zeros((target_array_size, 1, 1, 2))
        
        for i in range(0, target_array_size):
            if target_array[i]==0:
                new_target_array[i] = image_label
            else:
                new_target_array[i] = border_label
                
        return new_target_array
    
    def convert_indices(self, indices_dict):
        
        image_label = np.zeros((1,1,2))
        border_label = np.zeros((1,1,2))
        image_label[0,0,1] = 1
        border_label[0,0,0] = 1
        
        indices_dict['borders'] = border_label
        indices_dict['images'] = image_label
        
        return indices_dict
    
    def data_generator(self, directory, batch_size=256, shuffle_data=False):
        
        X, y = [], []
        n = 0
        
        image_label = np.zeros((1,1,2))
        border_label = np.zeros((1,1,2))
        image_label[0,0,1] = 1
        border_label[0,0,0] = 1
        
        classes = os.listdir(directory)
        
        borders_class_files = os.listdir(os.path.join(directory, classes[0]))
        images_class_files = os.listdir(os.path.join(directory, classes[1]))
        
        while True:
            
            for i in range(0, len(borders_class_files)):
                
                border_img = np.array(Image.open(os.path.join(directory, classes[0], borders_class_files[i])))/255
                image_img = np.array(Image.open(os.path.join(directory, classes[1], images_class_files[i])))/255
                
                if (border_img.shape == (32,32,3)) and (image_img.shape == (32,32,3)):
                    X.append(border_img)
                    y.append(border_label)
                    X.append(image_img)
                    y.append(image_label)
                    n += 1
                
                if n==batch_size//2:
                    if shuffle_data:
                        X, y = shuffle(X, y, random_state=0)
                    
                    yield (np.array(X), np.array(y))
                    X, y = [], []
                    n = 0
                    
                
    
    def train(self, train_dir, validation_dir, model_path, epochs=10, batch_size=256):
        
        assert 'images' in os.listdir(train_dir)
        assert 'borders' in os.listdir(train_dir)
        assert 'images' in os.listdir(validation_dir)
        assert 'borders' in os.listdir(validation_dir)
        
        train_boxes_dir = os.path.join(train_dir, 'images')  
        train_no_boxes_dir = os.path.join(train_dir, 'borders')  
        validation_boxes_dir = os.path.join(validation_dir, 'images')
        validation_no_boxes_dir = os.path.join(validation_dir, 'borders')  
        
        num_boxes_tr = len(os.listdir(train_boxes_dir))
        num_no_boxes_tr = len(os.listdir(train_no_boxes_dir))
        num_boxes_val = len(os.listdir(validation_boxes_dir))
        num_no_boxes_val = len(os.listdir(validation_no_boxes_dir))
        
        total_train = num_boxes_tr + num_no_boxes_tr
        total_val = num_boxes_val + num_no_boxes_val
        
        train_data_gen = self.data_generator(train_dir, batch_size=batch_size, shuffle_data=True)
        val_data_gen = self.data_generator(validation_dir, batch_size=batch_size, shuffle_data=False)
        
        self.__model_init(self.img_height, self.img_width)
        
        self.model.fit(
            train_data_gen,
            steps_per_epoch=total_train // batch_size,
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=total_val // batch_size
        )
            
        self.model.save_weights(os.path.join(model_path, "image_separator.h5"))
    
    def predict(self, input_images_path):
        
        pass

        
    def load_model(self, model_path, img_height=32, img_width=32):
        
        self.__model_init(img_height, img_width)
        
        self.model.load_weights(model_path)
    
    def trim(self, im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
    
    def create_mask(self, image, model_path):
        
        image = image.convert('RGB')
        
        image_processed = np.array(image)/255
        image_shape = image_processed.shape
        
        #self.load_model(model_path, image_shape[0], image_shape[1])
        #self.load_model(model_path, 32, 32)
        
        pred_array = self.model.predict_on_batch(image_processed.reshape(1,image_shape[0],image_shape[1],3))
        
        pred_array = np.around(pred_array)[0,:,:,:]
        pred_image = np.zeros((pred_array.shape[0], pred_array.shape[1]))
        
        for i in range(0, pred_image.shape[0]):
            for j in range(0, pred_image.shape[1]):
                if np.array_equal(pred_array[i][j], np.array([1., 0.])):
                    pred_image[i][j] = 255.
                else:
                    pred_image[i][j] = 0.
                    
        pred_image = Image.fromarray(pred_image.astype('uint8'))
        pred_image = pred_image.resize((image_shape[1], image_shape[0]), Image.NEAREST)
        
        return pred_image
    
    def get_image_parts(self, image_path, model_path, new_width=500):
        
        image = Image.open(image_path).convert('RGB')
        
        width, height = image.size
        new_height = int(new_width * height / width)
        
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        
        pred_image = self.create_mask(image, model_path)
        pred_image_array = np.array(pred_image)
        
        arr = pred_image_array > 0
        cleaned = morphology.remove_small_objects(arr, min_size=7500)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=7500)
        
        cleaned_image_mask = Image.fromarray(cleaned)
        cleaned_text_mask = Image.fromarray(np.invert(cleaned))
        
        final_image = Image.composite(cleaned_image_mask, image, cleaned_image_mask)
        final_text = Image.composite(cleaned_text_mask, image, cleaned_text_mask)
        
        final_image = self.trim(final_image)
        
        return final_image, final_text
    
if __name__ == "__main__":
    
    image_separator_path = '..//models'
    train_dir = '..//dataset//fake//train//sliding_window'
    validation_dir = '..//dataset//fake//validation//sliding_window'
    model_path = '..//models//image_separator.h5'
    text_saved = False
    
    image_separator = ImageSeparator()
    image_separator.load_model(model_path)
    #gen = image_separator.train_data_generator(train_dir, batch_size=4)
    #image_separator.train(train_dir, validation_dir, image_separator_path, epochs=5, batch_size=128)
    path = 'D://Projects//MemeGenerator//dataset//memes_filtered'
    images_out_path = 'D://Projects//MemeGenerator//dataset//final_images'
    texts_out_path = 'D://Projects//MemeGenerator//dataset//final_texts'
    last_image_path = 'D://Projects//MemeGenerator//dataset//last_image.txt'
    
    img_names = os.listdir(path)
    
    with open(last_image_path, 'r') as f:
                last_image = f.read()
    
    for i in tqdm(range(int(last_image),len(img_names))):
        img_path = os.path.join(path, img_names[i])
    
        final_image, final_text = image_separator.get_image_parts(img_path, model_path)
                
        try:
            with open(last_image_path, 'w', encoding='utf-8') as f:
                f.write(str(i))
            
            final_text.save(os.path.join(texts_out_path, img_names[i]))
            text_saved = True
            final_image.save(os.path.join(images_out_path, img_names[i]))
            
        except Exception as e:
            if text_saved:
                os.remove(os.path.join(texts_out_path, img_names[i]))
                text_saved = False
            print(f'Skipped {img_names[i]}: {e}')