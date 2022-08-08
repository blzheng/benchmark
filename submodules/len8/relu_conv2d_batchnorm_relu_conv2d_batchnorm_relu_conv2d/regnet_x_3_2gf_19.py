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
        self.relu66 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(432, 432, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=9, bias=False)
        self.batchnorm2d71 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU(inplace=True)
        self.conv2d72 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x228):
        x229=self.relu66(x228)
        x230=self.conv2d70(x229)
        x231=self.batchnorm2d70(x230)
        x232=self.relu67(x231)
        x233=self.conv2d71(x232)
        x234=self.batchnorm2d71(x233)
        x235=self.relu68(x234)
        x236=self.conv2d72(x235)
        return x236

m = M().eval()
x228 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x228)
end = time.time()
print(end-start)
