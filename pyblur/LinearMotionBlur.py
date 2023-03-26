# -*- coding: utf-8 -*-
import math
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import line
import cv2
from .LineDictionary import LineDictionary

lineLengths =[3,5,7,9]
lineTypes = ["full", "right", "left"]
lineDict = LineDictionary()


def LinearMotionBlur_random(img):
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes)) 
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    return LinearMotionBlur(img, lineLength, lineAngle, lineType)


def LinearMotionBlur(img, dim, angle, linetype='full'):
    if dim%2==0:
        dim=dim+1
    lineDict.createNxNLines(dim)
    imgarray = np.array(img)
    imgarray=cv2.cvtColor(imgarray, cv2.COLOR_RGB2BGR)
    kernel = LineKernel(dim, angle, linetype)
    convolved=cv2.filter2D(imgarray, -1, kernel)
    convolved = cv2.cvtColor(convolved, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(convolved)
    return img


def LineKernel(dim, angle, linetype):
    kernelwidth = dim
    kernelCenter = int(math.floor(dim/2))
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    angle=SanitizeAngleValue(angle)
    lineAnchors = lineDict.lines[angle]
    if(linetype == 'right'):
        lineAnchors[0] = kernelCenter
        lineAnchors[1] = kernelCenter
    if(linetype == 'left'):
        lineAnchors[2] = kernelCenter
        lineAnchors[3] = kernelCenter
    rr,cc = line(lineAnchors[0], lineAnchors[1], lineAnchors[2], lineAnchors[3])
    kernel[rr,cc]=1
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor        
    return kernel


def SanitizeAngleValue(angle):
    # numDistinctLines = kernelCenter * 4
    angle = math.fmod(angle, 180.0)
    validLineAngles = np.array([i for i in lineDict.lines])
    # print(validLineAngles)
    # validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angle = nearestValue(angle, validLineAngles)
    return angle


def nearestValue(theta, validAngles):
    idx = (np.abs(validAngles-theta)).argmin()
    # print(idx)
    return validAngles[idx]


def randomAngle(kerneldim):
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])





