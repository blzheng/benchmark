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
        self.batchnorm2d129 = BatchNorm2d(1696, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu129 = ReLU(inplace=True)
        self.conv2d129 = Conv2d(1696, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x456):
        x457=self.batchnorm2d129(x456)
        x458=self.relu129(x457)
        x459=self.conv2d129(x458)
        return x459

m = M().eval()
x456 = torch.randn(torch.Size([1, 1696, 14, 14]))
start = time.time()
output = m(x456)
end = time.time()
print(end-start)
