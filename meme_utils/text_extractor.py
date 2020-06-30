import re
import os
import nltk
from tqdm import tqdm
import pytesseract

import numpy as np
import pandas as pd

from PIL import Image
from collections import Counter 
from wordsegment import load, segment

load()

class TextExtractor:
    
    pytesseract.pytesseract.tesseract_cmd = (r"C:/Program Files/Tesseract-OCR/tesseract.exe")
    
    slash_n = ' nnnn '
    question = ' qqqq '
    
    separator = '<SEPR>'
    lf = '<N>'
    double_lf = '<NN>'
    
    additional_words = ['meme','memes','youre', '2017', '2018', '2019', '2020']
    
    startseq = 'when'
    endseq = 'endseq'
    
    WORDS = set(nltk.corpus.words.words())
    WORDS.update([slash_n, separator, question, 'nnnn', 'qqqq'])
    WORDS.update(additional_words)
    
    def __init__(self, images_path):
        
        self.images_path = images_path
        self.filenames = os.listdir(self.images_path)
        
    def extract_texts(self):
        texts = []
        
        for i in tqdm(range(0, len(self.filenames))):
            filename = os.path.join(self.images_path, self.filenames[i])
            texts.append(pytesseract.image_to_string(Image.open(filename), lang='eng'))
            
        return texts
    
    def __sentence_processing(self, sent):
        
        sent_segm = [segment(word) if word.isalpha() else word for word in sent.split(' ')]
        sent = ' '.join([' '.join(el) if type(el)==list else el for el in sent_segm])
        return ' '.join(word for word in nltk.wordpunct_tokenize(sent) if word.lower() in self.WORDS or not word.isalpha())
    
    def process_texts(self, texts):
        
        if len(texts) == 0:
            raise Exception('Nothing to process! You need to call extract_texts() first.')
        
        texts = [re.sub(r'[^a-z0-9?\n ]','', sent.lower()) for sent in texts] 
        texts = [sent.replace('\n', self.slash_n) for sent in texts]
        texts = [sent.replace('?', self.question) for sent in texts]

        split_it = ' '.join(texts).split() 
        counter = Counter(split_it) 
        most_occur = counter.most_common(59) 
        additional_words = np.squeeze(pd.DataFrame(most_occur)[[0]].to_numpy()).tolist()
        self.WORDS.update(additional_words)
        
        texts = [self.__sentence_processing(sent) for sent in texts]
        texts = [sent.replace((self.slash_n[:-1] * 4).strip(), self.separator)
                      .replace((self.slash_n[:-1] * 3).strip(), self.separator)
                      .replace((self.slash_n[:-1] * 2).strip(), self.double_lf)
                      .replace(self.question.strip(), '?')
                      .replace(self.slash_n[:-1].strip(), self.lf) for sent in texts]

        #texts = [f'{self.startseq} {sent}' if sent.split()[0] != self.startseq else sent for sent in texts]
        #texts = [f'{sent} {self.endseq}' for sent in texts]
        
        for i in tqdm(range(len(texts))):
            
            if texts[i] != '':
                if texts[i].split()[0] != self.startseq:
                    texts[i] = f'{self.startseq} {texts[i]}'
                
                texts[i] = f'{texts[i]} {self.endseq}'
                            
        
        df_image_texts = pd.DataFrame(list(zip(self.filenames, texts)), columns=['Image','Text'])
        
        no_text_indexes = df_image_texts[df_image_texts['Text']==''].index
        
        df_image_texts = df_image_texts.drop(no_text_indexes)
        
        return df_image_texts
    
images_path = '..//dataset//final_texts'
text_extractor = TextExtractor(images_path)

#texts = text_extractor.extract_texts()

df_image_texts = text_extractor.process_texts(texts)
