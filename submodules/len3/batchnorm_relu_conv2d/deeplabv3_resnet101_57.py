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
        self.batchnorm2d89 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu85 = ReLU(inplace=True)
        self.conv2d90 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x296):
        x297=self.batchnorm2d89(x296)
        x298=self.relu85(x297)
        x299=self.conv2d90(x298)
        return x299

m = M().eval()
x296 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x296)
end = time.time()
print(end-start)
