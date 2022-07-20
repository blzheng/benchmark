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
        self.conv2d43 = Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x146, x131):
        x147=operator.add(x146, x131)
        x148=self.conv2d43(x147)
        x149=self.batchnorm2d39(x148)
        return x149

m = M().eval()
x146 = torch.randn(torch.Size([1, 192, 14, 14]))
x131 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x146, x131)
end = time.time()
print(end-start)
