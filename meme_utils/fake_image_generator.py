import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from distutils.dir_util import copy_tree
from PIL import Image, ImageDraw, ImageFont




class FakeImageGenerator:
    
    text = '''
    ...he wondered if the Sorting Hat was genuinely conscious in the sense of being aware of its own awareness, and if so, whether it was satisfied with only getting to talk to eleven-year-olds once per year.
    Its song had implied so: Oh, I'm the Sorting Hat and I'm okay, I sleep all year and I work one day...
    
    When there was once more silence in the room, Harry sat on the stool and carefully placed onto his head the 800-year-old telepathic artefact of forgotten magic.
    
    Thinking, just as hard as he could: Don't Sort me yet! I have questions I need to ask you! Have I ever been Obliviated? Did you Sort the Dark Lord when he was a child and can you tell me about his weaknesses? Can you tell me why I got the brother wand to the Dark Lord's?
    Is the Dark Lord's ghost bound to my scar and is that why I get so angry sometimes? Those are the most important questions, but if you've got another moment can you tell me anything about how to rediscover the lost magics that created you?
    
    Into the silence of Harry's spirit where before there had never been any voice but one, there came a second and unfamiliar voice, sounding distinctly worried:
    
    "Oh, dear. This has never happened before..."
    
    What?
    
    "I seem to have become self-aware."
    
    WHAT?
    
    There was a wordless telepathic sigh. "Though I contain a substantial amount of memory and a small amount of independent processing power, my primary intelligence comes from borrowing the cognitive capacities of the children on whose heads I rest.
    I am in essence a sort of mirror by which children Sort themselves.
    But most children simply take for granted that a Hat is talking to them and do not wonder about how the Hat itself works, so that the mirror is not self-reflective.
    And in particular they are not explicitly wondering whether I am fully conscious in the sense of being aware of my own awareness."
    
    There was a pause while Harry absorbed all this.
    
    Oops.
    
    "Yes, quite. Frankly I do not enjoy being self-aware. It is unpleasant. It will be a relief to get off your head and cease to be conscious."
    
    But... isn't that dying?
    
    "I care nothing for life or death, only for Sorting the children. And before you even ask, they will not let you keep me on your head forever and it would kill you within days to do so."
    
    But - !
    
    "If you dislike creating conscious beings and then terminating them immediately, then I suggest that you never discuss this affair with anyone else. I'm sure you can imagine what would happen if you ran off and talked about it with all the other children waiting to be Sorted."
    
    If you're placed on the head of anyone who so much as thinks about the question of whether the Sorting Hat is aware of its own awareness -
    
    "Yes, yes. But the vast majority of eleven-year-olds who arrive at Hogwarts haven't read Godel, Escher, Bach. May I please consider you sworn to secrecy? That is why we are talking about this, instead of my just Sorting you."
    '''.replace('\n',' ').replace('.','').replace('"','')
    
    def __init__(self):
        
        pass
        
    def border_put_text(self, border, text, dw=0):
    
        border = Image.fromarray(border)
    
        border_shape = border.size
        text_size = np.random.randint(20,25)
        font_type = ImageFont.truetype('arial.ttf', text_size)
        max_symbols = int(border_shape[0]/9)
        draw = ImageDraw.Draw(border)

        white_space_start = 0
        white_space_end = border_shape[1] - text_size

        for i in range(white_space_start, white_space_end, text_size):
            random_text_start = np.random.randint(0,len(text) - max_symbols)
            random_text_end = np.random.randint(10, max_symbols)
            
            meme_text = text[random_text_start:random_text_start+random_text_end]
            draw.text(xy=(dw,i), text=meme_text, fill=(21,21,21), font=font_type)
        
        border = np.array(border).astype('uint8') 
            
        del draw
        
        return border
        
    def make_horizontal_border(self, image, dw, color):
        
        border = np.ones((image.shape[0], dw, 3)) * color
        
        return border.astype('uint8')
    
    def make_vertical_border(self, image, dh, color):
        
        border = np.ones((dh, image.shape[1], 3)) * color
        
        return border.astype('uint8')
                
    def append_text(self, image, color, text, dw, dh, side):
                
        if dw > 0:
            border_h = self.make_horizontal_border(image, dw, color)
            image = np.hstack((image,border_h)).astype('uint8')
            image = np.hstack((border_h,image)).astype('uint8')
            
        
        border_v = self.make_vertical_border(image, dh, color)
        border_v = self.border_put_text(border_v, text=text, dw=dw)

        if side == 'up':
            image = np.vstack((image,border_v)).astype('uint8')
        elif side == 'low':
            image = np.vstack((border_v,image)).astype('uint8')
        
        return Image.fromarray(image)
    
    def border_datagen(self, input_images_path, output_path, text, stride=32, max_images=10000):
        
        borders_folder = os.path.join(output_path, 'borders')
        images_folder = os.path.join(output_path, 'images')
        
        os.mkdir(borders_folder)
        os.mkdir(images_folder)
        
        images = os.listdir(input_images_path)
        
        for k in tqdm(range(0, len(images))):
            try:
                image_limit = False
                border_limit = False
                
                image = np.array(Image.open(os.path.join(input_images_path, images[k])))
                
                color = np.random.randint(240,256)
                dh = np.random.randint(int(image.shape[0] * 0.25),int(image.shape[0] * 0.4))
                
                if np.random.uniform() > 0.5:
                    dw = np.random.randint(int(image.shape[1] * 0.05),int(image.shape[1] * 0.1))
                else: 
                    dw = 0
                    
                border =  self.make_vertical_border(image, dh, color)
                border = self.border_put_text(border, text, dw=dw)
                
                image_shape = image.shape
                border_shape = border.shape
                
                new_image_res = ((image_shape[1]//stride)*stride, (image_shape[0]//stride)*stride)
                new_border_res = ((border_shape[1]//stride)*stride, (border_shape[0]//stride)*stride)
                
                image = Image.fromarray(image)
                border = Image.fromarray(border)
                
                image = image.resize(new_image_res, Image.ANTIALIAS)
                border = border.resize(new_border_res, Image.ANTIALIAS)
                
                for i in range(0, new_image_res[0], stride):
                    
                    if image_limit:
                        break
                    
                    for j in range(0, new_image_res[1], stride):
                        
                        if len(os.listdir(images_folder)) >= max_images:
                            image_limit = True
                            break    
                        
                        area = (i, j, i+stride, j+stride)
                        
                        image_crop = image.crop(area)
                        
                        image_crop.save(os.path.join(images_folder, f'{images[k][:-4]}_{str(i)}_{str(j)}.jpg'))
                        
                                                                        
            
                for i in range(0, new_border_res[0], stride):
                    
                    if border_limit:
                        break
                    
                    for j in range(0, new_border_res[1], stride):
                        
                        if len(os.listdir(borders_folder)) >= max_images:
                            border_limit = True
                            break 
                        
                        area = (i, j, i+stride, j+stride)
                        
                        border_crop = border.crop(area)
                        
                        border_crop.save(os.path.join(borders_folder, f'border_{images[k][:-4]}_{str(i)}_{str(j)}.jpg'))
                        
                        
                        
            except Exception as e:
                print(f'Skpipped {i}: {e}')
        
        
    def fake_datagen(self, input_images_path, output_path, text):
        
        boxes_folder = os.path.join(output_path, 'boxes')
        no_boxes_folder = os.path.join(output_path, 'no_boxes')
        
        os.mkdir(boxes_folder)
        os.mkdir(no_boxes_folder)
        
        copy_tree(input_images_path, no_boxes_folder)
        
        images = os.listdir(input_images_path)
        
        for i in range(0, len(images)):
            try:
                image = np.array(Image.open(os.path.join(input_images_path, images[i])))
        
                color = np.random.randint(240,256)
                dh_up = np.random.randint(int(image.shape[0] * 0.25),int(image.shape[0] * 0.4))
                dh_low = np.random.randint(int(image.shape[0] * 0.25),int(image.shape[0] * 0.4))
                
                if np.random.uniform() > 0.5:
                    dw = np.random.randint(int(image.shape[1] * 0.05),int(image.shape[1] * 0.1))
                else: 
                    dw = 0
        
                random_border_v = np.random.uniform()
        
                if random_border_v < 0.33:
                    image = self.append_text(image, color, self.text, dw, dh_up, side='up')
                    image = np.array(image)
                    image = self.append_text(image, color, self.text, dw, dh_low, side='low')
                elif 0.33 < random_border_v < 0.66:
                    image = self.append_text(image, color, self.text, dw, dh_up, side='up')
                else:
                    image = self.append_text(image, color, self.text, dw, dh_low, side='low')
                
                image.save(boxes_folder + '\\' + str(i) + '.jpg')
                
            except Exception as e:
                os.remove(no_boxes_folder + '\\' + images[i])
                print(f'Skpipped {i}: {e}')