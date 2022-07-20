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
        self.conv2d30 = Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d18 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x90):
        x91=self.conv2d30(x90)
        x92=self.batchnorm2d18(x91)
        return x92

m = M().eval()
x90 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x90)
end = time.time()
print(end-start)
