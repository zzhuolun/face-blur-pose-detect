import os
import cv2
import numpy as np
from hdface.hdface import hdface_detector
from skimage import transform as trans

__all__ = ['detect_face', 'detect_face_single']

src1 = np.array([
    [51.642, 50.115],
    [57.617, 49.990],
    [35.740, 69.007],
    [51.157, 89.050],
    [57.025, 89.702]], dtype=np.float32)
# <--left
src2 = np.array([
    [45.031, 50.118],
    [65.568, 50.872],
    [39.677, 68.111],
    [45.177, 86.190],
    [64.246, 86.758]], dtype=np.float32)

# ---frontal
src3 = np.array([
    [39.730, 51.138],
    [72.270, 51.138],
    [56.000, 68.493],
    [42.463, 87.010],
    [69.537, 87.010]], dtype=np.float32)

# -->right
src4 = np.array([
    [46.845, 50.872],
    [67.382, 50.118],
    [72.737, 68.111],
    [48.167, 86.758],
    [67.236, 86.190]], dtype=np.float32)

# -->right profile
src5 = np.array([
    [54.796, 49.990],
    [60.771, 50.115],
    [76.673, 69.007],
    [55.388, 89.702],
    [61.257, 89.050]], dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]], dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        # print(i,error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=50, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def square_bbox(bbox):
    """
        convert bbox to square
    Parameters:
    ----------
        bbox: numpy array , shape n x m
            input bbox
    Returns:
    -------
        square bbox
    """
    square_bbox = np.zeros(4, dtype='int')
    h = bbox[3] - bbox[1] + 1
    w = bbox[2] - bbox[0] + 1
    l = np.maximum(h, w)
    square_bbox[0] = int(bbox[0] + w * 0.5 - l * 0.5)
    square_bbox[1] = int(bbox[1] + h * 0.5 - l * 0.5)

    square_bbox[2] = square_bbox[0] + l - 1
    square_bbox[3] = square_bbox[1] + l - 1
    square_bbox[0] = np.maximum(int(bbox[0] + w * 0.5 - l * 0.5), 0)
    square_bbox[1] = np.maximum(int(bbox[1] + h * 0.5 - l * 0.5), 0)
    return square_bbox


def detect_face(img_dir, face_dir, img_name, face_name, crop_size=112, threshold=60, align=False):
    """"
    Args:
        img_dir: directory of to be detected images that contains one or more faces
        face_dir: detected faces will be written to this directory
        img_name: text file, includes all the image names in img_dir
        face_name: text file, includes all the face images names in face_dir
        crop_size: output face size
        threshold: if face_size < threshold, we won't accept it as a face
        align: bool, if True, do face alignment based on 5 landmark points
    face detection method: hdface
    output faces are square shape, i.e. the aspect ratio is 1
    """
    det = hdface_detector(use_cuda=True, min_face_size=80)
    f = open(img_name, 'r')
    face_name = open(face_name, 'w+')
    ll = f.readlines()
    for line in ll:
        try:
            path = os.path.join(img_dir, line)
            # path = img_dir + line
            img = cv2.imread(path[:-1])
            if img.shape[0] > 512 or img.shape[1] > 512:
                img = cv2.resize(img, (512, int(512 * float(img.shape[0]) / img.shape[1])),
                                 interpolation=cv2.INTER_AREA)  # in case that img is too large
            img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = det.detect_face(img_bg)
            if len(result) > 0:
                for i in range(len(result)):
                    box = result[i]['box']
                    if box[2] - box[0] < threshold or box[3] - box[1] < threshold:
                        continue
                    if align:
                        pts = result[i]['pts']
                        landmark = np.array([
                            [pts['leye'][0], pts['leye'][1]],
                            [pts['reye'][0], pts['reye'][1]],
                            [pts['nose'][0], pts['nose'][1]],
                            [pts['lmouse'][0], pts['lmouse'][1]],
                            [pts['rmouse'][0], pts['rmouse'][1]]
                        ])
                        face = norm_crop(img, landmark, crop_size, 'other')  #alignment
                    else:
                        squarebox = square_bbox(box)
                        squarebox[2] = np.minimum(squarebox[2], img.shape[1])
                        squarebox[3] = np.minimum(squarebox[3], img.shape[0])
                        face = img[squarebox[1]:squarebox[3] + 1, squarebox[0]:squarebox[2] + 1, :]
                        # add black border to face, if face not square, to avoid possible distortion of face
                        if face.shape[0] != face.shape[1]:
                            w = face.shape[1]
                            h = face.shape[0]
                            top = max((w - h) // 2, 0)
                            bottom = top
                            left = max((h - w) // 2, 0)
                            right = left
                            face = cv2.copyMakeBorder(face, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])

                    face_subdir = os.path.join(face_dir, line.split('/')[0])
                    if not os.path.exists(face_subdir):
                        os.mkdir(face_subdir)
                    imgwrite_dir = os.path.join(face_dir, line[:-5] + '_' + str(i) + '.jpg')
                    cv2.imwrite(imgwrite_dir, face)
                    face_name.write(line[:-5] + '_' + str(i) + '.jpg\n')
        except Exception as e:
            print(e)
    f.close()
    face_name.close()


#Following function is a variation of `detect_face` function:
#this function only find the biggest face in images at `img_dir` and write the single face to `face_dir`
def detect_face_single(img_dir, face_dir):
    det = hdface_detector(use_cuda=True, min_face_size=120)
    sub_dir_ls = os.listdir(img_dir)
    for sub_dir in sub_dir_ls:  # 01
        root_dir = os.path.join(img_dir, sub_dir)  # .../images/01/
        img_ls = os.listdir(root_dir)  # xxx.jpg
        for img_name in img_ls:
            if img_name.endswith('.jpg'):
                path = os.path.join(root_dir, img_name)  # .../images/01//xxx.png
                # path = img_dir + line
                img = cv2.imread(path)
                img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = det.detect_face(img_bg)
                if len(result) == 0:
                    print(path)
                    continue
                elif len(result) > 1:
                    img_square = []
                    for i in range(len(result)):
                        box = result[i]['box']
                        box_ = [i if i > 0 else 0 for i in box]
                        box_[2] = np.minimum(box_[2], img.shape[1])
                        box_[3] = np.minimum(box_[3], img.shape[0])
                        img_square.append(box_[2] + box_[3] - box_[0] - box_[1])
                    trueface_idx = img_square.index(max(img_square))
                else:
                    trueface_idx = 0
                box = result[trueface_idx]['box']
                squarebox = square_bbox(box)
                squarebox[2] = np.minimum(squarebox[2], img.shape[1])
                squarebox[3] = np.minimum(squarebox[3], img.shape[0])
                face = img[squarebox[1]:squarebox[3] + 1, squarebox[0]:squarebox[2] + 1, :]
                # add black border to face, if face not square, to avoid possible distortion of face
                if face.shape[0] != face.shape[1]:
                    w = face.shape[1]
                    h = face.shape[0]
                    top = max((w - h) // 2, 0)
                    bottom = top
                    left = max((h - w) // 2, 0)
                    right = left
                    face = cv2.copyMakeBorder(face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                face_subdir = os.path.join(face_dir, sub_dir)
                if not os.path.exists(face_subdir):
                    os.mkdir(face_subdir)
                imgwrite_dir = os.path.join(face_subdir, img_name)
                cv2.imwrite(imgwrite_dir, face)
