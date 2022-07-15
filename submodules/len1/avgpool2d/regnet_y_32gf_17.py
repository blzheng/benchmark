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
        self.adaptiveavgpool2d17 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x287):
        x288=self.adaptiveavgpool2d17(x287)
        return x288

m = M().eval()
x287 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x287)
end = time.time()
print(end-start)
