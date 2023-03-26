from __future__ import print_function, division
import warnings
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import *
from mobileNetv2 import mobilenet_v2
warnings.filterwarnings("ignore")
# plt.ion()  # interactive mode




# dir of testing face images
rootDir = ROOTDIR
# .txt file including names of images in rootDir
labelDir = LABELDIR
# to be tested model
model_dir = MODELDIR
# saving test resultes to save_dir directory
save_dir = SAVEDIR
size_x = 112
size_y = 112
batchSize = 4


class markDataset(Dataset):
    def __init__(self, label_dir, root_dir, transformImage):
        self.frame = pd.read_csv(label_dir)
        self.root_dir = root_dir
        self.transformImage = transformImage

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transformImage(image)  # after this step, type(image)= tensor
        sample = {'image': image, 'img_name': self.frame.iloc[idx, 0]}
        return sample


image_datasets = markDataset(label_dir=labelDir,
                             root_dir=rootDir,
                             transformImage=transforms.Compose([
                                 transforms.Resize((size_x, size_y)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ]))
dataloaders = DataLoader(image_datasets, batch_size=batchSize,
                         shuffle=True, num_workers=4)
dataset_sizes = len(image_datasets)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mobilenet_v2(pretrained=False, num_classes=1, width_mult=0.25)
state_dict = torch.load(model_dir)
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:]  # remove `module.`
#     new_state_dict[name] = v
model = nn.DataParallel(model, device_ids=[0, ])
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
cnt = 0
for data in dataloaders:
    inputs = data['image'].to(device)
    blurr, pose = model(inputs)
    for i in range(inputs.size()[0]):
        blur = blurr[i, 0].item() / 10.0
        blur = blur if blur < 1.0 else 1.0
        blur = round(blur, 2)
        yaw = int(pose[i, 0].item() * 10)
        pitch = int(pose[i, 1].item() * 10)
        roll = int(pose[i, 2].item() * 10)
        name = data['img_name'][i]
        # print(dict[name])
        img = cv2.imread(os.path.join(rootDir, name))
        img = plot_pose_cube(img, yaw, pitch, roll, size=img.shape[0] // 2)
        title = str(blur) + ' (' + str(yaw) + ',' + str(pitch) + ',' + str(roll) + ')'
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.imwrite(save_dir + name.split('.')[0] + '_' + str(blur) + '.jpg', img)
        cnt += 1
        if cnt == 10:
            cv2.destroyAllWindows()
            cnt = 0

        # f.write(data['img_name'][i] + ',' + str(round(blur, 2))
        #         +','+str(round(yaw,2)) +','+str(round(pitch, 2)) +',' + str(round(roll, 2))
        #         + '\n')
# f.close()
# view_dataset(rootDir, mark_txt)
