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
        self.relu12 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x37, x45):
        x46=operator.add(x37, x45)
        x47=self.relu12(x46)
        x48=self.conv2d15(x47)
        x49=self.batchnorm2d15(x48)
        return x49

m = M().eval()
x37 = torch.randn(torch.Size([1, 672, 28, 28]))
x45 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x37, x45)
end = time.time()
print(end-start)
