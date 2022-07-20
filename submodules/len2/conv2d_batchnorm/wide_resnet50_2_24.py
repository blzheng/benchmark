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
        self.conv2d24 = Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x78):
        x79=self.conv2d24(x78)
        x80=self.batchnorm2d24(x79)
        return x80

m = M().eval()
x78 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x78)
end = time.time()
print(end-start)
