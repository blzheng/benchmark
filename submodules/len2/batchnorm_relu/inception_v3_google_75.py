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
        self.batchnorm2d75 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x257):
        x258=self.batchnorm2d75(x257)
        x259=torch.nn.functional.relu(x258,inplace=True)
        return x259

m = M().eval()
x257 = torch.randn(torch.Size([1, 192, 5, 5]))
start = time.time()
output = m(x257)
end = time.time()
print(end-start)
