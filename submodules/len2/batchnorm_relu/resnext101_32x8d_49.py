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
        self.batchnorm2d76 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu73 = ReLU(inplace=True)

    def forward(self, x251):
        x252=self.batchnorm2d76(x251)
        x253=self.relu73(x252)
        return x253

m = M().eval()
x251 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x251)
end = time.time()
print(end-start)
