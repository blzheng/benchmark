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
        self.conv2d59 = Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
        self.batchnorm2d49 = BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x198):
        x199=self.conv2d59(x198)
        x200=self.batchnorm2d49(x199)
        return x200

m = M().eval()
x198 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x198)
end = time.time()
print(end-start)
