from __future__ import print_function, division
import copy
import time
import warnings
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils import *
from mobileNetv2 import *

warnings.filterwarnings("ignore")
plt.ion()  # interactive mode
rootDir = 'final_trainset/'  #root directory of all datasets
testFile = [os.path.join(rootDir, 'annotations/bs_wf_300wlp/test'), os.path.join(rootDir, 'annotations/bs_wf_biwi/test'),
            os.path.join(rootDir, 'annotations/bw_wf_300wlpFlip/test'), os.path.join(rootDir, 'annotations/bs_hpi_china/test'),
            os.path.join(rootDir, 'annotations/bs_wf_300wlpRot_biwiRot/test')]
trainFile = [os.path.join(rootDir, 'annotations/bs_wf_300wlp/train'), os.path.join(rootDir, 'annotations/bs_wf_biwi/train'),
             os.path.join(rootDir, 'annotations/bw_wf_300wlpFlip/train'), os.path.join(rootDir, 'annotations/bs_hpi_china/train'),
             os.path.join(rootDir, 'annotations/bs_wf_300wlpRot_biwiRot/train')]
dataset_num = len(trainFile)
model_load = 'blurposeMix_13.pt'
model_save = SAVEDIR

size_x = 112
size_y = 112
batchSize = 16

# since 300WLP dataset already provide flipped images and corresponding label, we don't do flip for 300WLP dataset
flip = [False, True, False, True, True]
dataloaders = []
dataset_sizes = []
for i in range(dataset_num):
    trainset = FaceDataset(csv_file=trainFile[i],
                    root_dir=rootDir,
                    resize=transforms.Resize([size_x, size_y]),
                    transform_all=transforms.Compose([
                        transforms.Resize([size_x, size_y]),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                    gaussianNoiseSigma=4, poissonNoiseFactor=1,
                    scale=10,
                    transform_blur=transforms.Compose([
                        transforms.RandomRotation(30, expand=True),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomAffine(0, (0.05, 0.05), scale=None, shear=[-5, 5], resample=False,
                                                fillcolor=0),
                    ]),
                    transform_pose=flip[i],
                    # addBlur=(75, 6, 26),
                    addBlur=(45, 6, 20),
                    )
    valset = FaceDataset(csv_file=testFile[i],
                    root_dir=rootDir,
                    resize=transforms.Resize([size_x, size_y]),
                    transform_all=transforms.Compose([
                        transforms.Resize([size_x, size_y]),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                    gaussianNoiseSigma=4, poissonNoiseFactor=1,
                    scale=10
                    )

    trainloader = DataLoader(trainset, batch_size=batchSize,
                             shuffle=True, num_workers=16)
    valloader = DataLoader(valset, batch_size=batchSize,
                           shuffle=True, num_workers=16)
    # # ------------------------------------------
    # def imshow(img):
    #     # img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    #
    # # get some random training images
    # dataiter = iter(trainloader)
    # for i in range(10):
    #     data = dataiter.next()
    #     images = data['image']
    #     # show images
    #     imshow(torchvision.utils.make_grid(images))
    #     # print labels
    #     print(data['blur'])
    #     print(data['euler'])
    #     print(data['tag'])
    #     print(i)

    image_datasets = {'train': trainset, 'val': valset}
    dataloaders.append({'train': trainloader, 'val': valloader})
    dataset_sizes.append({x: len(image_datasets[x]) for x in ['train', 'val']})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('dataset size:', dataset_sizes)


def train_model(model, dataLoader, datasetSize, num_epochs=25, best_acc=float('inf')):
    since = time.time()
    cnt = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = float('inf')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                cnt += 1
            else:
                model.eval()  # Set model to evaluate mode
            print(phase + ' mode')
            running_loss = 0.0
            for idx, data in enumerate(dataLoader[phase]):
                # ------ one batch --------
                inputs = data['image'].to(device)
                tag = data['tag']
                blurs = data['blur'].to(device)
                eulers = data['euler'].to(device)
                # track history if only in train
                blist = []
                plist = []
                for i in range(tag.size(0)):
                    if tag[i].item() == 0:
                        blist.append(i)
                    else:
                        plist.append(i)
                with torch.set_grad_enabled(phase == 'train'):
                    outblurs, outeulers = model.forward(inputs)
                    blurs = blurs[blist, :]
                    eulers = eulers[plist, :]
                    outblurs = outblurs[blist, :]
                    outeulers = outeulers[plist, :]
                    if phase == 'train':
                        bloss = criterion(outblurs, blurs)
                        ploss = criterion(outeulers, eulers)
                        optimizer.zero_grad()
                        al_loss = bloss + ploss
                        al_loss.backward()
                        optimizer.step()
                    else:
                        bloss = valCriterion(outblurs, blurs)
                        ploss = valCriterion(outeulers, eulers)
                        al_loss = bloss + ploss
                print(round,'round ',idx, 'batch complete:', bloss.item(), ploss.item(), al_loss.item())
                # statistics
                running_loss += al_loss.item() * inputs.size(0)
            epoch_loss = running_loss / datasetSize[phase]
            print('{} blur loss: {:.4f} '.format(
                phase, epoch_loss))
            if phase == 'val' and epoch_loss < best_acc:
                best_acc = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                cnt = 0
                torch.save(model.state_dict(), model_save) #save current best state dict
                print(epoch, 'updated')
        #if no improvement at val accuracy after 10 epoch, stop training
        if cnt > 10:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


net = mobilenet_v2(pretrained=False, num_classes=1, width_mult=0.25)
state_dict = torch.load(model_load)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
net = nn.DataParallel(net, device_ids=[0, 1, 2, ])
net.load_state_dict(state_dict)
net = net.to(device)
criterion = nn.MSELoss()
valCriterion = nn.L1Loss()
# Observe that all parameters are being optimized
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)

# accuracy = [float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]
accuracy = [0.83, 0.77, 0.93, 1.05, float('inf')] #history best val accuracy for differect datasets
for round in range(0,50):  #for each round, do 5 epochs training on one dataset
    print('-'*20,round,'th round','-'*20)
    print('accuracy:', accuracy)
    net, accuracy[round%dataset_num] = train_model(net, dataloaders[round%dataset_num], dataset_sizes[round%dataset_num],
                                         num_epochs=5, best_acc=accuracy[round%dataset_num])



