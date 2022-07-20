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
        self.conv2d38 = Conv2d(128, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.batchnorm2d38 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x136):
        x137=torch.nn.functional.relu(x136,inplace=True)
        x138=self.conv2d38(x137)
        x139=self.batchnorm2d38(x138)
        x140=torch.nn.functional.relu(x139,inplace=True)
        return x140

m = M().eval()
x136 = torch.randn(torch.Size([1, 128, 12, 12]))
start = time.time()
output = m(x136)
end = time.time()
print(end-start)
