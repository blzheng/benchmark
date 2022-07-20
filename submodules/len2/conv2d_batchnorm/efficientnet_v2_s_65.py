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
        self.conv2d95 = Conv2d(960, 960, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d65 = BatchNorm2d(960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x304):
        x305=self.conv2d95(x304)
        x306=self.batchnorm2d65(x305)
        return x306

m = M().eval()
x304 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x304)
end = time.time()
print(end-start)
