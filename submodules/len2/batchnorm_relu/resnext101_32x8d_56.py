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
        self.batchnorm2d86 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)

    def forward(self, x284):
        x285=self.batchnorm2d86(x284)
        x286=self.relu82(x285)
        return x286

m = M().eval()
x284 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x284)
end = time.time()
print(end-start)
