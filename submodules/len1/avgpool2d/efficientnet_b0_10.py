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
        self.adaptiveavgpool2d10 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x156):
        x157=self.adaptiveavgpool2d10(x156)
        return x157

m = M().eval()
x156 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
