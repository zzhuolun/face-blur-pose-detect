"""
This is the library, including some useful classes and functions in this project.
"""
from __future__ import print_function, division
import os
import random
from math import cos, sin
from random import randrange
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision import transforms
import pyblur


"""
--- Generate Gaussian or Poisson noise ---
image: read through PIL.Image.open('path')
sigma: variance of gaussian noise
factor: the bigger this value is, the more noisy is the poisson_noised image

##IMPORTANT: when reading a image into numpy arrary, the default dtype is uint8,
which can cause wrapping when adding noise onto the image. 
E.g,  example = np.array([128,240,255], dtype='uint8')
     example + 50 = np.array([178,44,49], dtype='uint8')
Transfer np.array to dtype='int16' can solve this problem.
"""
def gaussian_noise(image, sigma):
    img = np.array(image)
    noise = np.random.randn(img.shape[0], img.shape[1], img.shape[2])
    img = img.astype('int16')
    img_noise = img + noise * sigma
    img_noise = np.clip(img_noise, 0, 255)
    img_noise = img_noise.astype('uint8')
    return Image.fromarray(img_noise)


def poisson_noise(image, factor):
    factor = 1 / factor
    img = np.array(image)
    img = img.astype('int16')
    img_noise = np.random.poisson(img * factor) / float(factor)
    np.clip(img_noise, 0, 255, img_noise)
    img_noise = img_noise.astype('uint8')
    return Image.fromarray(img_noise)


"""
--- Dataset of pose and blur estimation ---
"""

class FaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, resize, transform_all,
                 gaussianNoiseSigma, poissonNoiseFactor, scale, transform_blur=None, transform_pose=False,
                 addBlur=None):
        """
        Args:
            csv_file (string): Path to the label text.
            root_dir (string): Directory with all the images.
            resize: torchvision.transforms.Resize, resize images in the dataset to the same size
            transform_all:transform to all images (despite blur or pose images)
            gaussianNoiseSigma: automatically add gaussian noise to blur images
            poissonNoiseFactor: automatically add poisson noise to blur images
            scale: scale the blur values, e.g. original blur range from 0-1, scale=10 means that blur value ranges from 0-10 now
            transform_blur: transform for blur images
            transform_pose: transform for pose images
            addBlur: range of automatically added blur. The values should be mannually decided.
        """
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.resize = resize
        self.transformBlur = transform_blur
        self.transformPose = transform_pose
        self.transformAll = transform_all
        self.gaussianNoiseSigma = gaussianNoiseSigma
        self.poissonNoiseFactor = poissonNoiseFactor
        self.addBlur = False
        self.scale = scale
        if addBlur:
            self.addBlur = True
            self.motionBlur = addBlur[0]
            self.gaussianBlur = addBlur[1]
            self.defocusBlur = addBlur[2]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[idx, 0])
        image = Image.open(img_name)
        tag = float(self.frame.iloc[idx, 1]) #0-blur image; 1-pose images.
        blur = float(self.frame.iloc[idx, 2])  # original blur value
        yaw = float(self.frame.iloc[idx, 3])
        pitch = float(self.frame.iloc[idx, 4])
        roll = float(self.frame.iloc[idx, 5])
        image = self.resize(image)
        """following code does data augmentation:
        1)randomly add blur to blur images;
        2)randomly flip pose images and rotate pose images with small yaw angle;
        3)randomly add noise to blur and pose images"""
        #randomly add blur
        if self.addBlur and randrange(1, 6) > 2 and blur > 0.83:
            blur_type = randrange(1, 6)
            if blur_type <= 3:  # motion blur, blur_type=1,2,3
                lineType = 'left' if randrange(0, 2) == 0 else 'right'
                kernelSize = randrange(3, self.motionBlur + 1)  # 3-75
                mid = 40.0
                a = np.power(2, 1 / mid)
                lamda = np.power(a, -kernelSize)
                blur = 0.94 * lamda   #change the original blur value
                image = pyblur.LinearMotionBlur(image,
                                                kernelSize,  # kernel size
                                                randrange(0, 16),  # angle
                                                lineType)
            elif blur_type == 4:  # gaussian blur, blur_type=4
                kernelSize = randrange(3, self.gaussianBlur + 1)  # 3-6
                mid = 3
                a = np.power(2, 1 / mid)
                lamda = np.power(a, -kernelSize + 1)
                blur = 0.94 * lamda #change the original blur value
                image = pyblur.GaussianBlur(image,
                                            kernelSize)
            else:  # defocus blur, blur_type=5
                kernelSize = randrange(3, self.defocusBlur + 1)  # 3-26
                mid = 9.0
                a = np.power(2, 1 / mid)
                lamda = np.power(a, -kernelSize + 2)
                blur = 0.94 * lamda #change the original blur value
                image = pyblur.DefocusBlur(image,
                                           kernelSize)
        #randomly add noise
        noise_type = randrange(1, 3)
        if noise_type == 1:  # add gaussian noise
            image = gaussian_noise(image, randrange(1, self.gaussianNoiseSigma + 1))
        else:  # apply poisson noise
            image = poisson_noise(image, randrange(1, self.poissonNoiseFactor + 1))

        if tag == 0 and self.transformBlur:
            image = self.transformBlur(image)  # only blur images in trainset do this transform
        #randomly flip or rotate pose-images
        if tag == 1:
            if self.transformPose and random.random() < 0.5:
                image = transforms.functional.hflip(image)  #randomly flip the image and change the euler angle values
                yaw = -yaw
                roll = -roll
            if abs(yaw) < 20 and random.random() < 0.5:  #only do rotate to images with a small yaw angle, in order to avoid affecting pitch angle while rotate the image
                if abs(yaw) <= 5 and abs(roll) < 45:
                    rot_deg = randrange(-45 - int(roll), 45 - int(roll))
                    image = transforms.functional.rotate(image, -rot_deg, False, False, None)
                    roll = roll + rot_deg
                if 5 < abs(yaw) <= 10 and abs(roll) < 30:
                    rot_deg = randrange(-30 - int(roll), 30 - int(roll))
                    image = transforms.functional.rotate(image, -rot_deg, False, False, None)
                    roll = roll + rot_deg
                if abs(yaw) > 10 and abs(roll) < 15:
                    rot_deg = randrange(-15 - int(roll), 15 - int(roll))
                    image = transforms.functional.rotate(image, -rot_deg, False, False, None)
                    roll = roll + rot_deg
            #randomly add small blur to pose-images
            if random.random() < 0.5:
                blur_type = randrange(0, 2)
                if blur_type == 0:
                    lineType = 'left' if randrange(0, 2) == 0 else 'right'
                    kernelSize = randrange(3, 8)  # 3-7
                    image = pyblur.LinearMotionBlur(image, kernelSize, randrange(0, 16), lineType)
                else:
                    image = pyblur.DefocusBlur(image, 3)

        image = self.transformAll(image)
        if tag == 0:
            blur = blur * self.scale
            if blur > self.scale:
                blur = self.scale
            if blur < 0:
                blur = 0
            blur = float(blur)
        else:
            yaw = yaw / 10
            roll = roll / 10
            pitch = pitch / 10
        tag = tensor([tag])
        blur = tensor([blur])
        euler = tensor([yaw, pitch, roll])
        sample = {'image': image, 'tag': tag, 'blur': blur, 'euler': euler}
        return sample


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(10)  # pause a bit so that plots are updated


# """This function views the dataset
# Parameters:
#     data_dir-- here stored the images in the dataset
#     label_dir -- this is the .txt file which store the image name and blurness, format is as follows:
#                     img_name,blurness
# """
#
#
# def view_dataset(data_dir, label_dir, sub_dir=None):
#     label = open(label_dir, 'r')
#     lbline = label.readlines()
#     cnt = 0
#     for line in lbline:
#         if sub_dir:
#             face_file = line.split('/')[0]
#             if not sub_dir == face_file:
#                 continue
#         ll = line.split(',')
#         try:
#             img = cv2.imread(data_dir + ll[0])
#             cv2.namedWindow(ll[0], cv2.WINDOW_NORMAL)
#             cv2.imshow(ll[1] + ',' + ll[2] + ',' + ll[3] + ',' + ll[4], img)
#             cv2.waitKey(0)
#             cnt += 1
#             if cnt == 10:
#                 cv2.destroyAllWindows()
#                 cnt = 0
#         except Exception as e:
#             print(e)
#             continue
#     cv2.destroyAllWindows()
#     label.close()


"""
This function extract names from a folder, and write  the file names in this folder to a .txt file
Parameters:
        rootdir -- directory of the folder
        name_txt -- directory of the output txt file
        one_level -- for folder with no subfolderes, one_level is true;
                    for folder with a level of subfolders, one_level is False 
"""


def extract_name(rootdir, name_txt, one_level=True):
    cnt = 0
    list = os.listdir(rootdir)
    f = open(name_txt, 'w+')
    if one_level:
        for i in range(len(list)):
            print(list[i])
            f.write(list[i] + '\n')
            cnt += 1
    else:
        for i in range(len(list)):
            subDir = os.listdir(os.path.join(rootdir, list[i]))
            # print(subDir)
            for j in range(len(subDir)):
                print(list[i] + '/' + subDir[j] + '\n')
                f.write(list[i] + '/' + subDir[j] + '\n')
                cnt += 1
    print(cnt, 'file found')


"""This function separate the label file into test label file and train label file
 Parameters:
     label_dir -- the total label file
     trainlabel_name -- from label_dir separated train label file
     testlabel_name -- from label_dir separeated test label file
     ratio -- number of train data/test data
 """


def separate_label(label_dir, trainlabel_name, testlabel_name, ratio=10):
    label = open(label_dir, 'r')
    label_test = open(testlabel_name, 'w+')
    label_train = open(trainlabel_name, 'w+')
    lbline = label.readlines()
    cnt = 0
    for line in lbline:
        try:
            if cnt % ratio == 0:  #
                label_test.write(line)
            else:
                label_train.write(line)
            cnt += 1
        except Exception as e:
            continue
    label.close()
    label_test.close()
    label_train.close()


"""
This function helps mannually lable the dataset blur values
"""
def manualLabel(data, lable_dir, new_lable_dir):
    with open(lable_dir, 'r') as label:
        row_number = sum(1 for _ in label)
    label = open(lable_dir, 'r')
    new_label = open(new_lable_dir, 'a')

    lbline = label.readlines()
    cnt = 0
    for line in lbline:
        ll = line.split(',')
        try:
            print("----", cnt, '/', row_number, "----")
            cnt += 1
            # if cnt<168:
            #     continue
            # if float(ll[1])>=0.9:
            #     if float(ll[1])>1.0:
            #         new_label.write(ll[0]+','+'1.0\n')
            #     else:
            #         new_label.write(line)
            # else:
            img = cv2.imread(data + ll[0])
            cv2.imshow(ll[1], img)  # + ' ' + str(img.shape[0]) + '*' + str(img.shape[1])
            cv2.waitKey(2000)
            key = input()
            if key == '+':
                new_label.write(line)
            elif key == '-':
                continue
            elif key == 'break':
                break
            else:
                blur = float(key)
                # print(ll[0]+','+blur+'\n')
                new_label.write(ll[0] + ',' + str(blur) + '\n')
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)
            continue
    cv2.destroyAllWindows()
    label.close()
    new_label.close()

#
# "Following code delete bad images in the image file of heils-face dataset"
#
# def cleanDataset(data, name, new_name, h_threshold, w_threshold):
#     faces = open(name, 'r')
#     new_label = open(new_name, 'w+')
#     lbline = faces.readlines()
#     cnt = 0
#     for line in lbline:
#         ll = line.split(',')
#         img = cv2.imread(data + ll[0])
#         try:
#             if img.shape[0] > h_threshold and img.shape[1] > w_threshold:
#                 # os.remove(data_dir+line[:-1])
#                 new_label.write(line)
#                 cnt += 1
#         except Exception as e:
#             continue
#     faces.close()
#     print(cnt, "total cleaned images")
#
#
# """
# This function moves all files in old_dir to new_dir with a new name(i.e prefix_num.jpg)
# """
#
#
# def renameImgfile(old_dir, new_dir, prefix):
#     list = os.listdir(old_dir)
#     cnt = 0
#     for idx, name in enumerate(list):
#         new_name = prefix + str(idx) + '.jpg'
#         if ' ' in name:
#             name = name[:name.index(' ')] + '\\' + name[name.index(' '):]
#         if ',' in name:
#             name = name[:name.index(',')] + '\\' + name[name.index(','):]
#         if '(' in name:
#             name = name[:name.index('(')] + '\\' + name[name.index('('):]
#         if ')' in name:
#             name = name[:name.index(')')] + '\\' + name[name.index(')'):]
#         if '&' in name:
#             name = name[:name.index('&')] + '\\' + name[name.index('&'):]
#         command = 'mv ' + old_dir + name + ' ' + new_dir + new_name
#         # print(command)
#         try:
#             os.system(command)
#             cnt += 1
#         except Exception as e:
#             print(e)
#             continue
#     print(len(list), 'file in old directory.')
#     print(cnt, 'files moved')


#
# def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
#     # yaw, pitch, roll : grb
#     pitch = pitch * np.pi / 180
#     yaw = -(yaw * np.pi / 180)
#     roll = roll * np.pi / 180
#
#     if tdx != None and tdy != None:
#         tdx = tdx
#         tdy = tdy
#     else:
#         height, width = img.shape[:2]
#         tdx = width / 2
#         tdy = height / 2
#
#     # X-Axis pointing to right. drawn in red
#     x1 = size * (cos(yaw) * cos(roll)) + tdx
#     y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
#
#     # Y-Axis | drawn in green
#     #        v
#     x2 = size * (-cos(yaw) * sin(roll)) + tdx
#     y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
#
#     # Z-Axis (out of the screen) drawn in blue
#     x3 = size * (sin(yaw)) + tdx
#     y3 = size * (-cos(yaw) * sin(pitch)) + tdy
#
#     cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
#     cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
#     cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)
#
#     return img


def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x1 - face_x), int(y2 + y1 - face_y)), (0, 0, 255), 3)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x2 - face_x), int(y1 + y2 - face_y)), (0, 0, 255), 3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x3 - face_x), int(y1 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x3 - face_x), int(y2 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (255, 0, 0), 2)
    # Draw top in green
    cv2.line(img, (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x1 - face_x), int(y3 + y1 - face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x2 - face_x), int(y3 + y2 - face_y)), (0, 255, 0), 2)

    return img
