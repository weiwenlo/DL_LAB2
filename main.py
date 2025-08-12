from oxford_pet import OxfordPetDataset
from PIL import Image
import matplotlib.pyplot as plt
# OxfordPetDataset.download("dataset/oxford-iiit-pet")
# print("Dataset downloaded and extracted!")

im = Image.open('dataset/oxford-iiit-pet/images/Abyssinian_1.jpg')
plt.imshow(im)
# plt.show()

import numpy as np

im_array = np.array(im)
print('Array Dimensions', im_array.shape)
print(im_array[0].shape)
print(im_array[0])