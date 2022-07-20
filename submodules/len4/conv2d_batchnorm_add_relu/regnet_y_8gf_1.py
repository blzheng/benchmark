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
        self.conv2d6 = Conv2d(224, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)

    def forward(self, x17, x5):
        x18=self.conv2d6(x17)
        x19=self.batchnorm2d4(x18)
        x20=operator.add(x5, x19)
        x21=self.relu4(x20)
        return x21

m = M().eval()
x17 = torch.randn(torch.Size([1, 224, 56, 56]))
x5 = torch.randn(torch.Size([1, 224, 56, 56]))
start = time.time()
output = m(x17, x5)
end = time.time()
print(end-start)
