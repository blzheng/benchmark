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
        self.batchnorm2d190 = BatchNorm2d(1760, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu190 = ReLU(inplace=True)

    def forward(self, x671):
        x672=self.batchnorm2d190(x671)
        x673=self.relu190(x672)
        return x673

m = M().eval()
x671 = torch.randn(torch.Size([1, 1760, 7, 7]))
start = time.time()
output = m(x671)
end = time.time()
print(end-start)
