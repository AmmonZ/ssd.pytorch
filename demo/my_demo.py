#构建测试阶段的SSD模型
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
#import torch.nn as nn
#import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from data import VOC_CLASSES as labels

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
net = build_ssd('test', 300, 2)    # initialize SSD 类别2
net.load_weights('../weights/ssd300_rice_25000.pth')

#加载没有标签的测试图片集
# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
img_id = 60 #第60个，注意这个60不是img的文件名，只是顺数第60张
image = testset.pull_image(img_id)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# View the sampled input image before transform
#plt.figure(figsize=(10,10))
plt.figure()
plt.imshow(rgb_image)

# plt.show()
#输入预处理
x = cv2.resize(image, (300, 300)).astype(np.float32)
#x -= (104.0, 117.0, 123.0)
x -= (170.48119520399305, 181.27863504774305, 174.67430427517363)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
plt.figure()
plt.imshow(x)

x = torch.from_numpy(x).permute(2, 0, 1)
#SSD前向传递
#把图片封装在Variable中，从而能够被pytorch autograd功能识别
xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)
#解析检测和观察结果
top_k=10

#plt.figure(figsize=(10,10))
plt.figure()
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib

currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1

plt.show()