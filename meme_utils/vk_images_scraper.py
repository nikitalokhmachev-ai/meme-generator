import time
import wget

from selenium import webdriver
from selenium.webdriver import ActionChains


class VKImagesScraper:
    
    DRIVER_PATH = 'D:/ChromeDriver/chromedriver.exe'
    
    def __init__(self, group, output_path):
        
        self.group = group
        self.output_path = output_path
        self.images_links = []
        
    def scroll_to_end(self, wd):
        
        self.wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
    def group_to_url(self, group):
        
        base_url = 'https://vk.com/'
        group_url = base_url + group
        
        return group_url
        
    def extract_links(self, number_images=None):
        
        images_counter = 0
        scroll_counter = 0
        skips_counter = 0
        
        group_url = self.group_to_url(self.group)
        
        if number_images is not None:
            number_scrolls = round(number_images/10+0.5)
        else:
            number_scrolls = None
        
        self.wd = webdriver.Chrome(executable_path=self.DRIVER_PATH)
        self.wd.get(group_url)
                
        print('Press Ctrl+C to stop scrolling.')
        
        try:
            while True:
                self.scroll_to_end(self.wd)
                scroll_counter += 1
                time.sleep(0.5)
                
                if number_scrolls is not None:
                    if scroll_counter >= number_scrolls:
                        break
                
        except KeyboardInterrupt:
            print('Scrolling stopped.')
        
        content = self.wd.find_elements_by_class_name('wall_text')
        
        print('Press Ctrl+C to stop extracting data.')
        
        try:
            for i in range(0,len(content)):
                
                try:
                    ActionChains(self.wd).click(content[i].find_elements_by_tag_name('a')[0]).perform()
                    time.sleep(0.3)
                
                    link = self.wd.find_element_by_id('pv_photo').find_element_by_tag_name('img').get_attribute('src')
                    images_counter += 1
                    
                    print('Got {}: {}'.format(images_counter, link))
                    self.images_links.append(link)
                    self.wd.execute_script("window.history.go(-1)")
                    
                    if images_counter > number_images:
                        break
                    
                    skips_counter = 0
                    
                except Exception as e:
                    skips_counter += 1
                    print('Skipped {}: {}'.format(images_counter, e))
                    if skips_counter > 10:
                        self.wd.execute_script("window.history.go(1)")
                        
        except KeyboardInterrupt:
            print('Extracting data stopped.')
            
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
