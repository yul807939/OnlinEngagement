import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter
import utils


Path_images = "./BIWI/"
Path_labels = "./dataset/BIWI/"
rs = 1
Data_Split = [11 , 12 ,1  , 21 , 8 , 19 , 20]


def ExtractRotateMatrix(path):
    n = []
    with open(path, 'r') as file:
        for i in range(1 , 4):
            line = file.readline().split(" ")
            a = [float(x) for x in line[0:3]]
            n.append(a)
    file.close()
    n = np.array(n)
    n = np.transpose(n)
    return n

def rotateMatrixToEulerAngles(R):  
    roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
    yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
    pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

    out_euler = [yaw , pitch , roll]
        
    return out_euler


def get_list_from_filenames(file_path):

    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


class BIWI(Dataset):
    def __init__(self, filename_path, transform, image_mode='RGB', train_mode=False,):
        
        self.transform = transform

        d = np.load(filename_path , allow_pickle=True)
        if filename_path[-7 : -4] == "AEF":
            x_data = d['image']
            y_data = d['pose']
        else:
            x_data = d['images']
            y_data = d['labels']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.X_train[index]))
        img = img.convert(self.image_mode)

        roll = self.y_train[index][2]/180*np.pi
        yaw = self.y_train[index][0]/180*np.pi
        pitch = self.y_train[index][1]/180*np.pi
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        return img, cont_labels, torch.FloatTensor(R), self.X_train[index]

    def __len__(self):
        return self.length

class Pose_300W_LP(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(
            self.data_dir+ "/" + self.X_train[index][2:] + self.img_ext)
        img = img.convert(self.image_mode)
        mat_path = os.path.join(
            self.data_dir + "/" + self.y_train[index][2:] + self.annot_ext)

        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        pose = utils.get_ypr_from_mat(mat_path)
        pitch = pose[0] 
        yaw = pose[1] 
        roll = pose[2]


        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        pose_labels = torch.FloatTensor((yaw, pitch, roll))
        if self.transform is not None:
            img = self.transform(img)

        return img,  pose_labels,[], self.X_train[index]

    def __len__(self):
        return self.length




