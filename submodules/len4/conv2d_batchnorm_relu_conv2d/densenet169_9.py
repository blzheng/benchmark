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
        self.conv2d20 = Conv2d(224, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x75):
        x76=self.conv2d20(x75)
        x77=self.batchnorm2d21(x76)
        x78=self.relu21(x77)
        x79=self.conv2d21(x78)
        return x79

m = M().eval()
x75 = torch.randn(torch.Size([1, 224, 28, 28]))
start = time.time()
output = m(x75)
end = time.time()
print(end-start)
