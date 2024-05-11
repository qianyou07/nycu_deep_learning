import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import cv2


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.320, 0.223, 0.160], std=[0.304, 0.219, 0.175]),
        ])
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""
        path = self.root + self.img_name[index] + '.jpeg'
        label = self.label[index]
        img = cv2.imread(path)

        # Filter out black background
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)

        # Find out the minimum rectangle that bounds the foreground
        x, y, w, h = cv2.boundingRect(thresh)
        img = img[y:y+h, x:x+w]
        s = max(img.shape[0:2])

        # Pad rectangle with black color so that it becomes square
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = ImageOps.pad(img, (s, s), color='black')

        # resize to (512, 512)
        img = img.resize((512, 512))

        # transform
        img = self.transform(img)
        """
        step1. Get the image path from 'self.img_name' and load it.
                hint : path = root + self.img_name[index] + '.jpeg'
        
        step2. Get the ground truth label from self.label
                    
        step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                    
                In the testing phase, if you have a normalization process during the training phase, you only need 
                to normalize the data. 
                
                hints : Convert the pixel value to [0, 1]
                        Transpose the image shape from [H, W, C] to [C, H, W]
                        
            step4. Return processed image and label
        """

        return img, label
