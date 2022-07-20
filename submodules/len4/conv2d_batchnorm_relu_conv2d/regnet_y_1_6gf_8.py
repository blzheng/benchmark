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
        self.conv2d38 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=5, bias=False)

    def forward(self, x119):
        x120=self.conv2d38(x119)
        x121=self.batchnorm2d24(x120)
        x122=self.relu29(x121)
        x123=self.conv2d39(x122)
        return x123

m = M().eval()
x119 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x119)
end = time.time()
print(end-start)
