from PIL import Image
import numpy as np
import csv
import os
from keras import backend as K
import matplotlib.pyplot as plt

def convert_to_jpg(path):
    for i in range(322):
        if i < 9:
            img_path = _imgPath+'00'+str(i+1)
        elif i<99:
            img_path = _imgPath+'0'+str(i+1)
        else:
            img_path = _imgPath+str(i+1)

        img = Image.open(img_path+'.pgm')
        img.save(img_path+".jpg")

# get images as numpy array

def get_numpy_imgs(path, img_size):
    images = []
    for i in range(322):
    #values 00 and 0 are added to paths to coordinate to the file names in the folder bc_photos
    #ex mdb001, mdb10, mdb100
        if i < 9:
            temp_path = path+'00'+str(i+1)
        elif i<99:
            temp_path = path +'0' +str(i+1)
        else:
            temp_path = path +str(i+1)
    
        img = Image.open(temp_path+'.jpg')
        img.load()
        img = img.resize((img_size, img_size), Image.ANTIALIAS)
        data = np.asarray(img,dtype='float32')        
        images.append(data)
    
    result = np.array(images)
    return result

def get_flatened_images(path):
    not_flat = get_numpy_imgs(path)
    list_flat = []
    
    for i in range(322):
        temp = not_flat[i].flatten()
        list_flat.append(temp)
    
    return np.array(list_flat)

def gen_labels(file_name):
    labels = []
    txt_file = csv.reader(open(file_name), delimiter=" ")
    #if a mammogram returns as NORMAL, assign a value 0, if it is Benign(B) or Malignant(M) assign value 1 or 2
    for s in txt_file:
        if s[2] =='NORM':
            labels.append(int(0))
        #elif s[3] == 'B':
        #    labels.append(int(1))
        else:
            labels.append(int(1))
    
    np_labels = np.array(labels)
    #np.eye makes a value 3 to [0,0,1,0,0], essential one-hot encoding    
    #return np.eye(3)[np_labels]
    return np_labels

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()    
    
