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
        self.adaptiveavgpool2d59 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x1059):
        x1060=self.adaptiveavgpool2d59(x1059)
        return x1060

m = M().eval()
x1059 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1059)
end = time.time()
print(end-start)
