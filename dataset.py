"""Implements a dataset class for handling image data"""
from utils.image_utils import imread, imsave, imresize

#DATA_PATH_BASE = '/home/VoxelFlow/dataset/ucf101_triplets/'
#DATA_PATH_BASE = '/home/alex/video/dataset/DeepVideoDeblurring_Dataset/quantitative_datasets/'

#DATA_PATH_BASE = '/home/alex/video/dataset/UCF101/testcutmachine/frames/'
class Dataset(object):
  def __init__(self, data_list_file=None, process_func=None, DATA_PATH_BASE=""):
    """
      Args:
    """
    self.data_list_file = data_list_file
    self.DATA_PATH_BASE = DATA_PATH_BASE
    if process_func:
      self.process_func = process_func
    else:
      self.process_func = self.process_func
 
  def read_data_list_file(self):
    """Reads the data list_file into python list
    """
    f = open(self.data_list_file)
    data_list =  [self.DATA_PATH_BASE+line.rstrip() for line in f]
    self.data_list = data_list
    return data_list

  def process_func(self, example_line):
    """Process the single example line and return data 
      Default behavior, assumes each line is the path to a single image.
      This is used to train a VAE.
    """
    #return imresize(imread(example_line), [256, 256])
    return imread(example_line)
