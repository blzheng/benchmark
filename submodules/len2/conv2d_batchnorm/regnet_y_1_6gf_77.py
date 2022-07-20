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
        self.conv2d125 = Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=14, bias=False)
        self.batchnorm2d77 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x396):
        x397=self.conv2d125(x396)
        x398=self.batchnorm2d77(x397)
        return x398

m = M().eval()
x396 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x396)
end = time.time()
print(end-start)
