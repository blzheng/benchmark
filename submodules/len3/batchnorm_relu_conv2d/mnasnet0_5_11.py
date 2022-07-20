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
        self.batchnorm2d16 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x46):
        x47=self.batchnorm2d16(x46)
        x48=self.relu11(x47)
        x49=self.conv2d17(x48)
        return x49

m = M().eval()
x46 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x46)
end = time.time()
print(end-start)
