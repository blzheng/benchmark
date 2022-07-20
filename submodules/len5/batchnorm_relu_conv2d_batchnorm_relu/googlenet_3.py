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
        self.batchnorm2d10 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d11 = Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x48):
        x49=self.batchnorm2d10(x48)
        x50=torch.nn.functional.relu(x49,inplace=True)
        x51=self.conv2d11(x50)
        x52=self.batchnorm2d11(x51)
        x53=torch.nn.functional.relu(x52,inplace=True)
        return x53

m = M().eval()
x48 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x48)
end = time.time()
print(end-start)
