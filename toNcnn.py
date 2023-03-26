"""
This code snippet does the conversion from pyTorch model to ncnn model.
Convert pipeline: pyTorch model -> onnx model -> simplified onnx model -> ncnn model
See Ncnn wiki for detail:
https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx/7e5b62ba190a69b81a0d38a3949bbbe3c7bc54d0
"""
import os
from collections import OrderedDict
import torch.onnx
from PIL import Image
from torchvision import transforms
from mobileNetv2 import mobilenet_v2

model_load = MODELLOAD #to be converted pytorch model
onnx_save = ONNXSAVE
simonnx_save = SIMONNXSAVE

#destination ncnn model
ncnn_param='blurposeMix.param'
ncnn_bin = 'blurposeMix.bin'

state_dict = torch.load(model_load)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
model = mobilenet_v2(pretrained=False, num_classes=1, width_mult= 0.25)
model.load_state_dict(new_state_dict)
model.eval()

transform = transforms.Compose([transforms.Resize((112, 112)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
img = Image.open("posetest/faces/img_19._0.jpg")
input = transform(img).unsqueeze(0)

output = model(input)
print(output)
#pytorch model to onnx model
onnx_out=torch.onnx.export(model,  # model being run
                  input,  # model input (or a tuple for multiple inputs)
                  onnx_save,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  verbose=False,
                  input_names=['input'],  # the model's input names
                  output_names=['output'],
                  )
print(onnx_out) #you can also use onnxruntime to verify the output of onnx model
#simplify onnx model
os.system('python3 -m onnxsim '+onnx_save+' '+simonnx_save)
#simplified onnx model to ncnn model
os.system('ncnn/build/tools/onnx/onnx2ncnn '+simonnx_save+' '+ncnn_param+
          ' '+ncnn_bin)



