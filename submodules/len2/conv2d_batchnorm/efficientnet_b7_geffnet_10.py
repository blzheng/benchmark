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
        self.conv2d18 = Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d10 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x57):
        x58=self.conv2d18(x57)
        x59=self.batchnorm2d10(x58)
        return x59

m = M().eval()
x57 = torch.randn(torch.Size([1, 192, 112, 112]))
start = time.time()
output = m(x57)
end = time.time()
print(end-start)
