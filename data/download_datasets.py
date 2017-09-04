from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
import zipfile

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


