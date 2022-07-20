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
        self.conv2d44 = Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x161):
        x162=self.conv2d44(x161)
        x163=self.batchnorm2d44(x162)
        return x163

m = M().eval()
x161 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x161)
end = time.time()
print(end-start)
