from torch.autograd import Variable
#from utils.visualize import  make_dot
from ssd import build_ssd
import torch
from torchviz import make_dot

ssd_net = build_ssd('train', 300, 2)
net = ssd_net

x = Variable(torch.randn(1,3,300,300))

y = net(x)

dot = make_dot(y, params=dict(net.named_parameters()))
print(dot)
dot.view()