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
        self.conv2d12 = Conv2d(64, 160, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d12 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)

    def forward(self, x37, x47):
        x38=self.conv2d12(x37)
        x39=self.batchnorm2d12(x38)
        x48=operator.add(x39, x47)
        x49=self.relu12(x48)
        return x49

m = M().eval()
x37 = torch.randn(torch.Size([1, 64, 28, 28]))
x47 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x37, x47)
end = time.time()
print(end-start)
