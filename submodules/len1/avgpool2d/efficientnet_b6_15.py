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
        self.adaptiveavgpool2d15 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x234):
        x235=self.adaptiveavgpool2d15(x234)
        return x235

m = M().eval()
x234 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x234)
end = time.time()
print(end-start)