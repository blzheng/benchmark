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
        self.conv2d45 = Conv2d(352, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)

    def forward(self, x282):
        x283=self.conv2d45(x282)
        x284=self.batchnorm2d45(x283)
        x285=self.relu29(x284)
        return x285

m = M().eval()
x282 = torch.randn(torch.Size([1, 352, 7, 7]))
start = time.time()
output = m(x282)
end = time.time()
print(end-start)
