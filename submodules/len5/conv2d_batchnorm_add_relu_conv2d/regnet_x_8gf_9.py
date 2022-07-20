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
        self.conv2d24 = Conv2d(240, 720, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d24 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x77, x87):
        x78=self.conv2d24(x77)
        x79=self.batchnorm2d24(x78)
        x88=operator.add(x79, x87)
        x89=self.relu24(x88)
        x90=self.conv2d28(x89)
        return x90

m = M().eval()
x77 = torch.randn(torch.Size([1, 240, 28, 28]))
x87 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x77, x87)
end = time.time()
print(end-start)
