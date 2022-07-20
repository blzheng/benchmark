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
        self.batchnorm2d58 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu55 = ReLU(inplace=True)

    def forward(self, x190):
        x191=self.batchnorm2d58(x190)
        x192=self.relu55(x191)
        return x192

m = M().eval()
x190 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x190)
end = time.time()
print(end-start)
