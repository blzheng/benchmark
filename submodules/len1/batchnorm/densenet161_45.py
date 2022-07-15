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
        self.batchnorm2d45 = BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x162):
        x163=self.batchnorm2d45(x162)
        return x163

m = M().eval()
x162 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x162)
end = time.time()
print(end-start)
