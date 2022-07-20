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
        self.relu30 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d46 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x148):
        x149=self.relu30(x148)
        x150=self.conv2d46(x149)
        x151=self.batchnorm2d46(x150)
        return x151

m = M().eval()
x148 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x148)
end = time.time()
print(end-start)
