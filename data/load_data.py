import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

# import _pickle as pickle
import pickle
import numpy as np
import struct
from glob import glob

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_datasets(dataset:str, one_hot=False):

    if not dataset:
        return 'Dataset is not defined or invalid.'
    
    print(f'\nLoad data: {dataset}')
    if 'mnist' == dataset.lower():
        # MNIST
        inputs = dict()
        for f in glob("../application_data/mnist/*"): 
            key = os.path.split(f)[1]
            names = key.split('.')[0]
            inputs[names] = f
        datasets = load_mnist(inputs, one_hot)
        return datasets['train-images'], datasets['train-labels'], datasets['t10k-images'], datasets['t10k-labels']
    if 'fashionmnist' == dataset.lower():
        # Fashion MNIST
        inputs = dict()
        for f in glob("../application_data/fashionmnist/*"): 
            key = os.path.split(f)[1]
            names = key.split('.')[0]
            inputs[names] = f
            
        datasets = load_fashionmnist(inputs, one_hot)
        
        return datasets['train-images'], datasets['train-labels'], datasets['t10k-images'], datasets['t10k-labels']
    elif 'cifar10' == dataset.lower():
        # CIFAR-10
        inputs = dict()
        for f in glob("../application_data/cifar-10-batches-py/*"): 
            key = os.path.split(f)[1]
            inputs[key] = f
        return load_cifar10(inputs, one_hot)

    

def load_mnist(input_dict, one_hot=False):
    '''
    Load MNIST dataset 
    
    Args:
        input_dict: retrieved dictionary of files with dtype _io.BytesIO
        one_hot: apply one-hot encoding to labels
    
    Returns:
        output_dict: output dictionary contains loaded data arrays
    '''
    
    output_dict = dict()
    for key, file in input_dict.items():
        if 'images' in file.lower():
            with open(file, 'rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                nrows, ncols = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
                data = data.reshape((size, nrows, ncols))
        elif 'labels' in file.lower():
            with open(file, 'rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
            
            if one_hot:
                labels = np.zeros((len(data), 10))
                for i in range(len(labels)):
                    target = int(data[i])
                    labels[i, :][target] = 1
                data = labels
        else:
            data = []
        output_dict[key.split('.')[0]] = data
    return output_dict

def load_fashionmnist(input_dict, one_hot=False):
    '''
    Load Fashion MNIST dataset 
    
    Args:
        input_dict: retrieved dictionary of files with dtype _io.BytesIO
        one_hot: apply one-hot encoding to labels
    
    Returns:
        output_dict: output dictionary contains loaded data arrays
    '''
    
    output_dict = dict()
    for key, file in input_dict.items():
        if 'images' in file.lower():
            with open(file, 'rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                nrows, ncols = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
                data = data.reshape((size, nrows, ncols))
                output_dict[key.split('.')[0]] = data
        elif 'labels' in file.lower():
            with open(file, 'rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
            
            if one_hot:
                labels = np.zeros((len(data), 10))
                for i in range(len(labels)):
                    target = int(data[i])
                    labels[i, :][target] = 1
                data = labels
            output_dict[key.split('.')[0]] = data
        
    return output_dict

def load_cifar10(inputs, one_hot=False):
    '''
    Load CIFAR10 dataset using NAS API.
    
    Args:
        input_dict: retrieved dictionary of files with dtype _io.BytesIO
        one_hot: apply one-hot encoding to labels
    
    Returns:
        output_dict: output dictionary contains loaded data arrays
    '''
    
    #### CIFAR-10 constants
    img_size = 32
    img_channels = 3
    nb_classes = 10
    # length of the image after we flatten the image into a 1-D array
    img_size_flat = img_size * img_size * img_channels
    nb_files_train = 5
    images_per_file = 10000
    # number of all the images in the training dataset
    nb_images_train = nb_files_train * images_per_file
    
    def load_data(data_path, file_name):
        print('Loading ' + file_name)
        data = unpickle(data_path[file_name])
        raw_images = data[b'data']
        cls = np.array(data[b'labels'])
        images = raw_images.reshape([-1, img_channels, img_size, img_size])    
        # move the channel dimension to the last
        images = np.rollaxis(images, 1, 4)

        return images, cls

    def load_training_data(data_path):    
        # pre-allocate the arrays for the images and class-numbers for efficiency.
        images = np.zeros(shape=[nb_images_train, img_size, img_size, img_channels], 
                          dtype=np.uint8)
        cls = np.zeros(shape=[nb_images_train], dtype=np.uint8)

        begin = 0
        for i in range(nb_files_train):
            images_batch, cls_batch = load_data(data_path, file_name="data_batch_" + str(i + 1))
            num_images = len(images_batch)
            end = begin + num_images
            images[begin:end, :] = images_batch
            cls[begin:end] = cls_batch
            begin = end
        labels = utils.to_categorical(cls, nb_classes) if one_hot else cls
        return images, labels

    def load_test_data(data_path):
        images, cls = load_data(data_path, file_name="test_batch")
        labels = utils.to_categorical(cls, nb_classes) if one_hot else cls
        return images, labels
    
    X_train, y_train = load_training_data(inputs)
    X_test, y_test = load_test_data(inputs)
    
    return X_train, y_train, X_test, y_test
