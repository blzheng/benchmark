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
        self.batchnorm2d66 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x352):
        x353=self.batchnorm2d66(x352)
        return x353

m = M().eval()
x352 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x352)
end = time.time()
print(end-start)
