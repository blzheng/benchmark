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
        self.adaptiveavgpool2d21 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x327):
        x328=self.adaptiveavgpool2d21(x327)
        return x328

m = M().eval()
x327 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x327)
end = time.time()
print(end-start)
