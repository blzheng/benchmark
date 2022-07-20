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
        self.batchnorm2d53 = BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)

    def forward(self, x268):
        x269=self.batchnorm2d53(x268)
        x270=self.relu65(x269)
        return x270

m = M().eval()
x268 = torch.randn(torch.Size([1, 2016, 14, 14]))
start = time.time()
output = m(x268)
end = time.time()
print(end-start)
