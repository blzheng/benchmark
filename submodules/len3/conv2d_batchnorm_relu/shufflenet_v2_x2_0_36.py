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
        self.conv2d55 = Conv2d(976, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)

    def forward(self, x362):
        x363=self.conv2d55(x362)
        x364=self.batchnorm2d55(x363)
        x365=self.relu36(x364)
        return x365

m = M().eval()
x362 = torch.randn(torch.Size([1, 976, 7, 7]))
start = time.time()
output = m(x362)
end = time.time()
print(end-start)
