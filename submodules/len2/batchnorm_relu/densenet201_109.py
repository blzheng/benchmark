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
        self.batchnorm2d109 = BatchNorm2d(1376, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu109 = ReLU(inplace=True)

    def forward(self, x386):
        x387=self.batchnorm2d109(x386)
        x388=self.relu109(x387)
        return x388

m = M().eval()
x386 = torch.randn(torch.Size([1, 1376, 14, 14]))
start = time.time()
output = m(x386)
end = time.time()
print(end-start)
