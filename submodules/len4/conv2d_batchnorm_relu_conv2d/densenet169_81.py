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
        self.conv2d166 = Conv2d(1632, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d167 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu167 = ReLU(inplace=True)
        self.conv2d167 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x589):
        x590=self.conv2d166(x589)
        x591=self.batchnorm2d167(x590)
        x592=self.relu167(x591)
        x593=self.conv2d167(x592)
        return x593

m = M().eval()
x589 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x589)
end = time.time()
print(end-start)
