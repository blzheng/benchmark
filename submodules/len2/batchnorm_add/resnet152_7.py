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
        self.batchnorm2d20 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x65, x58):
        x66=self.batchnorm2d20(x65)
        x67=operator.add(x66, x58)
        return x67

m = M().eval()
x65 = torch.randn(torch.Size([1, 512, 28, 28]))
x58 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x65, x58)
end = time.time()
print(end-start)
