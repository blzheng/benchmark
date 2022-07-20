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
        self.batchnorm2d99 = BatchNorm2d(1216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu99 = ReLU(inplace=True)

    def forward(self, x351):
        x352=self.batchnorm2d99(x351)
        x353=self.relu99(x352)
        return x353

m = M().eval()
x351 = torch.randn(torch.Size([1, 1216, 14, 14]))
start = time.time()
output = m(x351)
end = time.time()
print(end-start)
