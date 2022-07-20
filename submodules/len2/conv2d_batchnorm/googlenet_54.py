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
        self.conv2d54 = Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x186):
        x196=self.conv2d54(x186)
        x197=self.batchnorm2d54(x196)
        return x197

m = M().eval()
x186 = torch.randn(torch.Size([1, 832, 7, 7]))
start = time.time()
output = m(x186)
end = time.time()
print(end-start)
