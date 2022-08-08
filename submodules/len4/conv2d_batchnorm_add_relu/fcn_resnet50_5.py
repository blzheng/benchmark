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
        self.conv2d14 = Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d14 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)

    def forward(self, x38, x46):
        x47=self.conv2d14(x38)
        x48=self.batchnorm2d14(x47)
        x49=operator.add(x46, x48)
        x50=self.relu10(x49)
        return x50

m = M().eval()
x38 = torch.randn(torch.Size([1, 256, 56, 56]))
x46 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x38, x46)
end = time.time()
print(end-start)
