import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.relu45 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x149, x157):
        x158=operator.add(x149, x157)
        x159=self.relu45(x158)
        x160=self.conv2d49(x159)
        x161=self.batchnorm2d49(x160)
        return x161

m = M().eval()
x149 = torch.randn(torch.Size([1, 720, 14, 14]))
x157 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x149, x157)
end = time.time()
print(end-start)
