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
        self.adaptiveavgpool2d3 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x66):
        x67=self.adaptiveavgpool2d3(x66)
        return x67

m = M().eval()
x66 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x66)
end = time.time()
print(end-start)
