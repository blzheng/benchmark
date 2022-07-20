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
        self.batchnorm2d37 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)

    def forward(self, x122):
        x123=self.batchnorm2d37(x122)
        x124=self.relu34(x123)
        return x124

m = M().eval()
x122 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x122)
end = time.time()
print(end-start)
