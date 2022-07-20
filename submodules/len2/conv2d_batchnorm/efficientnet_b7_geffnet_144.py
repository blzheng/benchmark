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
        self.conv2d242 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d144 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x724):
        x725=self.conv2d242(x724)
        x726=self.batchnorm2d144(x725)
        return x726

m = M().eval()
x724 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x724)
end = time.time()
print(end-start)
