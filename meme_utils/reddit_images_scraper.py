from psaw import PushshiftAPI
from PIL import Image
import wget
import os
import hashlib
from tqdm import tqdm

class RedditImagesScraper:
    
    def __init__(self, subreddit, output_path):
        
        self.subreddit = subreddit
        self.output_path = output_path
        self.images_links = []
        self.api = PushshiftAPI()
        self.new_memes = self.api.search_submissions(subreddit=self.subreddit)
        
    def extract_links(self, number_images=None):
        
        self.images_links = []
        counter = 0
        
        print('Scraping images... Press Ctrl+C to stop the process.')
        
        try:
            while True:
                
                try:
                    img = next(self.new_memes)
                    
                except:
                    print('No more images.')
                    break
                
                url = img.url
                
                if '.jpg' in url:
                    
                    self.images_links.append(url)
                    counter += 1
                    print(f'Images scraped: {counter}')
                
                if number_images is not None:
                    if counter >= number_images:
                        break
                    
        except KeyboardInterrupt:
            print('Extracting data stopped.')
            
        return self.images_links

    def verify_images(self):
        
        print('Verifying images...')
        
        for filename in tqdm(os.listdir(self.output_path)):
            
            try:
                img_path = self.output_path + '//' + filename
                img = Image.open(img_path)
                img.verify() 
              
            except:
                os.remove(img_path)
                print(f'{filename} removed!')
                
    def remove_duplicate_images(self):
        
        duplicates = []
        hash_keys = []
        
        file_list = os.listdir(self.output_path)
        
        print('Removing duplicate images...')
        
        for filename in tqdm(file_list):
           
            filename = os.path.join(self.output_path, filename)
            
            if os.path.isfile(filename):
                
                with open(filename, 'rb') as f:
                    filehash = hashlib.md5(f.read()).hexdigest()
                    
                if filehash not in hash_keys:
                    hash_keys.append(filehash)
                    
                else:
                    duplicates.append(filename)
            
        for duplicate in duplicates:
            
            os.remove(duplicate)
            print(f'{duplicate} removed!')
            
    def download_images(self):
        
        c = 0

        if len(self.images_links) != 0: 
            
            images_links_filtered = list(dict.fromkeys(self.images_links)) #remove repetitive elements
            
            print('Downloading...')
            
            for link in images_links_filtered:
                try:
                    wget.download(link, self.output_path + '\\' + str(c) + '.jpg')
                    print(link)
                    c += 1
                    
                except Exception as e:
                    print('Exception in image {}: {}. The link is: {}'.format(c, e, link))
        
        self.verify_images()
        self.remove_duplicate_images()