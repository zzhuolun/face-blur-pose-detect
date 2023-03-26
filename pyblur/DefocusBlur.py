import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import circle

defocusKernelDims = [3, 5, 7, 9]


def DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))
    kerneldim = defocusKernelDims[kernelidx]
    return DefocusBlur(img, kerneldim)


def DefocusBlur(img, dim):   #dim: kernel size
    imgarray = np.array(img, dtype="float32")
    kernel = DiskKernel(dim)
    if imgarray.ndim == 3 and imgarray.shape[-1] == 3:
        convolved = np.stack([convolve2d(imgarray[..., channel_id],
                                         kernel, mode='same',
                                         boundary='symm').astype("uint8")
                              for channel_id in range(3)], axis=2)
    else:
        convolved = convolve2d(imgarray, kernel, mode='symm', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img


def DiskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    circleCenterCoord = int(dim / 2)
    circleRadius = dim-circleCenterCoord

    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    # print(rr)
    # print(cc)
    kernel[rr, cc] = 1

    if (dim == 3 or dim == 5):
        kernel = Adjust(kernel, dim)

    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel


def Adjust(kernel, kernelwidth):
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel