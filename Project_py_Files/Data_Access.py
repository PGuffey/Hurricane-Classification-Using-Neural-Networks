import requests as re
from bs4 import BeautifulSoup as bs
import os
import time
import progressbar
import numpy as np
import configparser
import pathlib

#Import server url and download location pathname from configuration file
config = configparser.ConfigParser()
config.read('config.ini')

server_url = config['Server']['server_url']
download_location = config['Data']['download_location']

#Get list of nested urls from server
def get_urls(server_url:str)->list or None:
    data = re.get(server_url, allow_redirects=True)

    #Only raise exception if status code is a 4xx error
    if data.status_code != 200:
        data.raise_for_status()
        raise RuntimeError(f"Request to {server_url} returned status code {data.status_code}")

    names = []
    data = bs(data.content, 'html.parser')

    links = data.find_all('a')

    #Get html href content from link for url text
    filenames = [link.get('href') for link in links if link.get('href')]

    for filename in filenames:
        names.append(filename)

    return names

def get_file_urls(links:list, server_url:str)->list:
    urls = []
    #only download if file has a .gz extension
    for link in links:
        url = ''.join([server_url, link])
        headers = re.head(url).headers
        content_type = headers.get('Content-Type')
        if content_type == 'application/gzip':
            urls.append(url)
        else:
            pass
    return urls

def download_url_data(urls:list, download_location:str, n_chunk:int=1)->None:
    for url in urls:
        print(f"Downloading data at {url}")
        try:
            data = re.get(url, stream=True)
            if data.status_code != 200:
                data.raise_for_status()  # Will only raise for 4xx codes
                raise RuntimeError(f"Request to {server_url} returned status code {data.status_code}")

            #Stream 1kb blocks from file content and initialize progress bar from content length / chunk size = 1kb
            block_size = 1024
            file_size = int(data.headers.get('Content-Length', None))
            num_bars = np.ceil(file_size / (n_chunk * block_size))
            bar = progressbar.ProgressBar(maxval=num_bars).start()

            #Create file if url filename not already in download location
            file_name = os.path.basename(url)
            file_path = os.path.join(download_location, file_name)
            if pathlib.Path(file_path).is_file():
                print("Data already downloaded!")
                continue 
            else:
                os.makedirs(download_location, exist_ok=True)

            #Write chunk stream into file and update progress bar progressively
            with open(file_path, 'wb') as f:
                for iter, chunk in enumerate(data.iter_content(chunk_size=n_chunk * block_size)):
                    f.write(chunk)
                    bar.update(iter+1)
                    time.sleep(0.05)
            print(f"Downloaded {file_name} successfully.")
        
        #Catch file writing exceptions
        except Exception as e:
            print(f"Error downloading from {url}: {e}")
        
filenames = get_urls(server_url)
urls = get_file_urls(filenames, server_url)
download_url_data(urls, download_location)

