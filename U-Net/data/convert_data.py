"""
Source:
https://github.com/Slugskickass/Teaching_python/blob/master/Images/1.)%20Load%20and%20Save.py

"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def loadtiffs(file_name):
    img = Image.open(file_name)
    print('The Image is', img.size, 'Pixels.')
    print('With', img.n_frames, 'frames.')

    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    for I in range(img.n_frames):
        img.seek(I)
        imgArray[:, :, I] = np.asarray(img)
    img.close()
    return(imgArray)

def savetiffs(file_name, data):
    images = []
    for I in range(np.shape(data)[0]):
        images.append(Image.fromarray(data[I]))
    images[0].save(file_name, save_all=True, append_images=images[1:])
        #For a single image
        #images.save(file_name)

#Use the functions
file_name = 'epi.tif'
data_images = loadtiffs(file_name)
#data_images = np.transpose(data_images, (2,0,1)) #format needed for U-net

normalised = []
maps = []
thresholds = [0.28,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.35,0.12,0.12,0.3,0.3,0.4,0.4,0.4,0.4,0.4,0.4,0.4]

for i in range(20):
    im = (data_images[:,:,i] - np.min(data_images[:,:,i]))/(np.max(data_images[:,:,i])-np.min(data_images[:,:,i]))
    normalised.append(im)
    
    im2 = (im > thresholds[i])
    maps.append(im2)

savetiffs('train_images.tif', np.array(normalised))
savetiffs('train_maps.tif', np.array(maps))
