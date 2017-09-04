from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
import zipfile
import requests
import sys 

def download_PF_willow(dest="datasets"):
    if not exists(dest):
        makedirs(dest)
        url = "http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)
        
        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()
        
        remove(file_path)
        
        url = "http://www.di.ens.fr/willow/research/cnngeometric/other_resources/test_pairs_pf.csv"
        print("downloading url ", url)
        
        data = urllib.request.urlopen(url)
        file_path = join(dest, 'PF-dataset',basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

def download_pascal(dest="datasets/pascal-voc11"):
    if not exists(dest):
        makedirs(dest)
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)
        
        file_path = join(dest, basename(url))
        with open(file_path, "wb") as f:
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None: # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                    sys.stdout.flush()


        print("Extracting data")
        zip_ref = tarfile.open(file_path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()
        
        remove(file_path)
        
