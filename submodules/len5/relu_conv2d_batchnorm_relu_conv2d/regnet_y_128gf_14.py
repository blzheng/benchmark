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
        self.relu68 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU(inplace=True)
        self.conv2d90 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)

    def forward(self, x280):
        x281=self.relu68(x280)
        x282=self.conv2d89(x281)
        x283=self.batchnorm2d55(x282)
        x284=self.relu69(x283)
        x285=self.conv2d90(x284)
        return x285

m = M().eval()
x280 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x280)
end = time.time()
print(end-start)
