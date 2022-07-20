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
        self.batchnorm2d46 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x232, x219):
        x233=self.batchnorm2d46(x232)
        x234=operator.add(x219, x233)
        x235=self.relu56(x234)
        x236=self.conv2d75(x235)
        return x236

m = M().eval()
x232 = torch.randn(torch.Size([1, 440, 7, 7]))
x219 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x232, x219)
end = time.time()
print(end-start)
