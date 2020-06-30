from meme_utils.caption_generator import CaptionGenerator
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
from nltk.corpus import brown

class MemeCaptioner:
    
    def __init__(self, checkpoint_path='models/captiongen_ckpt', tokenizer_path='models/tokenizer.pickle', top_k=5000, max_length=190):
        
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        
        self.cg = CaptionGenerator(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path)
        self.nouns = np.array(list({word for word, pos in brown.tagged_words() if pos.startswith('NN')}))
    
    def make_horizontal_border(self, image, dw, color):
        
        border = np.ones((image.shape[0], dw, 3)) * color
        
        return border.astype('uint8')
    
    def make_vertical_border(self, image, dh, color):
        
        border = np.ones((dh, image.shape[1], 3)) * color
        
        return border.astype('uint8')
    
    def put_n(self, caption, max_symbols):
    
        new_caption = ''

        for i in range(0,len(caption),max_symbols):
            caption_part = caption[i:i+max_symbols]
            if '\n' not in caption_part:
                space_index = caption_part.rfind(' ')
                caption_part = caption_part[:space_index] + '\n' + caption_part[space_index+1:]

            new_caption += caption_part

        return new_caption
    
    def prepare_caption_text(self, caption, max_symbols=32):
        
        caption = caption.replace(' <n> ', '\n').replace('<n> ', '\n').replace(' <n>', '\n').replace('<sepr>', '\n')
        two_sides = False
        
        caption = self.put_n(caption, max_symbols)
        
        if '<unk>' in caption:
            random_noun = np.random.choice(self.nouns, 1)[0]
            caption = caption.replace('<unk>', random_noun)
        
        if '<nn>' in caption:
            
            caption_list = caption.split('<nn>',1)
            caption_list[1] = caption_list[1].replace(' <nn> ','\n').replace('<nn> ','\n').replace(' <nn>','\n')
            
            two_sides = True
            
            return caption_list, two_sides
        
        else:
            return caption, two_sides
    
    def border_put_text(self, border, text, text_size, dw=0):
    
        border = Image.fromarray(border)
        
        font_type = ImageFont.truetype('arial.ttf', text_size)
        draw = ImageDraw.Draw(border)

        draw.text(xy=(dw,0), text=text, fill=(21,21,21), font=font_type)
        
        border = np.array(border).astype('uint8') 
            
        del draw
        
        return border
    
    def append_text(self, image, color, text, text_size, dw, dh, side):
                  
        border_v = self.make_vertical_border(image, dh, color)
        border_v = self.border_put_text(border_v, text=text, text_size=text_size, dw=dw)

        if side == 'low':
            image = np.vstack((image,border_v)).astype('uint8')
        elif side == 'up':
            image = np.vstack((border_v,image)).astype('uint8')
        
        return image
    
    def make_caption(self, image, caption, text_size):
        
        channel = np.random.randint(240,256)
        color = (channel,channel,channel)
        
        wpercent = (500/float(image.size[0]))
        hsize = int((float(image.size[1])*float(wpercent)))
        image = image.resize((500,hsize), Image.ANTIALIAS)
        image = np.array(image)
        
        caption, two_sides = self.prepare_caption_text(caption)
        
        if np.random.uniform() > 0.5:
            dw = np.random.randint(int(image.shape[1] * 0.03),int(image.shape[1] * 0.07))
        else: 
            dw = 0
             
        if dw > 0:
            border_h = self.make_horizontal_border(image, dw, color)
            image = np.hstack((image,border_h)).astype('uint8')
            image = np.hstack((border_h,image)).astype('uint8')
            
        if two_sides:
            dh = (caption[0].count('\n')+1)*text_size + 10
            image = self.append_text(image, color, caption[0], text_size, dw, dh, side='up')
            dh = (caption[1].count('\n')+1)*text_size + 10
            image = self.append_text(image, color, caption[1], text_size, dw, dh, side='low')
        else:
            dh = (caption.count('\n')+1)*text_size + 10
            image = self.append_text(image, color, caption, text_size, dw, dh, side='up')
        
        return Image.fromarray(image), caption
    
    def generate_caption(self, file_url_path, byte_file=False):
        
        if not byte_file:
        
            if 'http' in file_url_path:
                image_tensor, response = self.cg.load_image_from_url(file_url_path)
                caption = self.cg.evaluate_from_img(image_tensor)
    
                image = Image.open(BytesIO(response.content)).convert('RGB')
                image.load()
                
            else:
                image_tensor = self.cg.load_image(file_url_path)
                caption = self.cg.evaluate_from_img(image_tensor)
    
                image = Image.open(file_url_path).convert('RGB')
                
        else:
            image_tensor = self.cg.encode(file_url_path)
            caption = self.cg.evaluate_from_img(image_tensor)
            
            image = Image.open(BytesIO(file_url_path)).convert('RGB')
            image.load()
        
        image, caption = self.make_caption(image, caption=caption, text_size=30)
        
        return image, caption