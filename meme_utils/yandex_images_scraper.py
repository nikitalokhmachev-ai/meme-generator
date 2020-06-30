import time
import wget
import re

from selenium import webdriver


class YandexImagesExtractor:
    
    DRIVER_PATH = 'D:/ChromeDriver/chromedriver.exe'
    
    def __init__(self, query, output_path):
        
        self.query = query
        self.output_path = output_path
        self.images_links = []
        
    def scroll_to_end(self, wd):
        
        self.wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5) 
        
    def query_to_url(self, query):
        
        base_url = 'https://yandex.ru/images/search?text='
        query_processed = re.sub(r'[^a-zа-я0-9 ]', '', query.lower()).replace(' ', '+')
        query_url = base_url + query_processed
        
        return query_url
        
    def extract_links(self, number_images=None):
        
        images_counter = 0
        enough_images = False
        query_url = self.query_to_url(self.query)
        
        self.wd = webdriver.Chrome(executable_path=self.DRIVER_PATH)
        self.wd.get(query_url)
        
        print('Press Ctrl+C to cancel extraction.')
        try:
            while not enough_images:
                
                self.scroll_to_end(self.wd)
                content = self.wd.find_elements_by_class_name('serp-item__link')
                links_to_image = [el.get_attribute('href') for el in content]
                
                for i in range(0,len(links_to_image)):
                    
                    self.wd.get(links_to_image[i])
                    
                    try:
                        open_button = self.wd.find_element_by_class_name('MMImage-Origin')
                        self.images_links.append(open_button.get_attribute('src'))
                        
                        images_counter += 1
                        
                        if number_images is not None:
                            if images_counter >= number_images:
                                enough_images = True
                                break
                    except:
                        print('skipped')
                
        except KeyboardInterrupt:
            print('Links extraction stopped.')
            
        return self.images_links
                
    def download_images(self):
        
        c = 0

        if len(self.images_links) != 0: 
            
            images_links_filtered = list(dict.fromkeys(self.images_links)) #remove repetitive elements
            
            print('Downloading...')
            
            for link in images_links_filtered:
                print(link)
                try:
                    wget.download(link, self.output_path + '\\' + str(c) + '.jpg')
                    c += 1
                except Exception as e:
                    print('Exception in image {}: {}'.format(c, e))
                    