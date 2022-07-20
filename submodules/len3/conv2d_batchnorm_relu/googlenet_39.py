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
        self.conv2d39 = Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x145):
        x146=self.conv2d39(x145)
        x147=self.batchnorm2d39(x146)
        x148=torch.nn.functional.relu(x147,inplace=True)
        return x148

m = M().eval()
x145 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x145)
end = time.time()
print(end-start)
