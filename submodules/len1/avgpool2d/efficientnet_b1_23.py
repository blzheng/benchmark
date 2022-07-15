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
        self.adaptiveavgpool2d23 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x354):
        x355=self.adaptiveavgpool2d23(x354)
        return x355

m = M().eval()
x354 = torch.randn(torch.Size([1, 1280, 7, 7]))
start = time.time()
output = m(x354)
end = time.time()
print(end-start)
