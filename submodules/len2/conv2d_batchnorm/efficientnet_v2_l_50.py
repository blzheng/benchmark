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
        self.conv2d62 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x207):
        x208=self.conv2d62(x207)
        x209=self.batchnorm2d50(x208)
        return x209

m = M().eval()
x207 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x207)
end = time.time()
print(end-start)
