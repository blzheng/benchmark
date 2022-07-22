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
        self.relu67 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d71 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d72 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d72 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x232):
        x233=self.relu67(x232)
        x234=self.conv2d71(x233)
        x235=self.batchnorm2d71(x234)
        x236=self.relu67(x235)
        x237=self.conv2d72(x236)
        x238=self.batchnorm2d72(x237)
        return x238

m = M().eval()
x232 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x232)
end = time.time()
print(end-start)
