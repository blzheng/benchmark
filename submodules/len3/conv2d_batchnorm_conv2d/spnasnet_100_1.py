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
        self.conv2d14 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d15 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x46):
        x47=self.conv2d14(x46)
        x48=self.batchnorm2d14(x47)
        x49=self.conv2d15(x48)
        return x49

m = M().eval()
x46 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x46)
end = time.time()
print(end-start)
