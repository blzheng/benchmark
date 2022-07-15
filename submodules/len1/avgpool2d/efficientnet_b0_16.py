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
        self.adaptiveavgpool2d16 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x245):
        x246=self.adaptiveavgpool2d16(x245)
        return x246

m = M().eval()
x245 = torch.randn(torch.Size([1, 1280, 7, 7]))
start = time.time()
output = m(x245)
end = time.time()
print(end-start)
