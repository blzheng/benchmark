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
        self.batchnorm2d14 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x44, x37):
        x45=self.batchnorm2d14(x44)
        x46=operator.add(x37, x45)
        x47=self.relu12(x46)
        x48=self.conv2d15(x47)
        return x48

m = M().eval()
x44 = torch.randn(torch.Size([1, 192, 28, 28]))
x37 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x44, x37)
end = time.time()
print(end-start)
