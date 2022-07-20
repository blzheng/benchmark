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
        self.adaptiveavgpool2d2 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x153):
        x154=self.adaptiveavgpool2d2(x153)
        return x154

m = M().eval()
x153 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)