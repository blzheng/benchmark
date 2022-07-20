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
        self.batchnorm2d21 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x68, x77):
        x69=self.batchnorm2d21(x68)
        x78=operator.add(x69, x77)
        x79=self.relu21(x78)
        x80=self.conv2d25(x79)
        return x80

m = M().eval()
x68 = torch.randn(torch.Size([1, 408, 14, 14]))
x77 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x68, x77)
end = time.time()
print(end-start)
