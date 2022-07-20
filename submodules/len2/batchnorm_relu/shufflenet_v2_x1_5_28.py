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
        self.batchnorm2d43 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)

    def forward(self, x278):
        x279=self.batchnorm2d43(x278)
        x280=self.relu28(x279)
        return x280

m = M().eval()
x278 = torch.randn(torch.Size([1, 352, 14, 14]))
start = time.time()
output = m(x278)
end = time.time()
print(end-start)
