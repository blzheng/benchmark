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
        self.batchnorm2d46 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)

    def forward(self, x165):
        x166=self.batchnorm2d46(x165)
        x167=self.relu46(x166)
        return x167

m = M().eval()
x165 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x165)
end = time.time()
print(end-start)
