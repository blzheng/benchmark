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
        self.conv2d131 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d77 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x393):
        x394=self.conv2d131(x393)
        x395=self.batchnorm2d77(x394)
        return x395

m = M().eval()
x393 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x393)
end = time.time()
print(end-start)
