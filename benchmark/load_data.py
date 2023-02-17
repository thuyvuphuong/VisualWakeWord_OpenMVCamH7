import wget
import tarfile

url = 'https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz'
filename = wget.download(url)

# open file
file = tarfile.open(filename)
file.extract('vw_coco2014_96')
file.close()
