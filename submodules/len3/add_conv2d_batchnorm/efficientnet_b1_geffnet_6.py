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
        self.conv2d54 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x159, x145):
        x160=operator.add(x159, x145)
        x161=self.conv2d54(x160)
        x162=self.batchnorm2d32(x161)
        return x162

m = M().eval()
x159 = torch.randn(torch.Size([1, 80, 14, 14]))
x145 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x159, x145)
end = time.time()
print(end-start)
