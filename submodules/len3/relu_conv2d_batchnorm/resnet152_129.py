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
        self.relu130 = ReLU(inplace=True)
        self.conv2d134 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d134 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x442):
        x443=self.relu130(x442)
        x444=self.conv2d134(x443)
        x445=self.batchnorm2d134(x444)
        return x445

m = M().eval()
x442 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x442)
end = time.time()
print(end-start)
