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
        self.adaptiveavgpool2d6 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x110):
        x111=self.adaptiveavgpool2d6(x110)
        return x111

m = M().eval()
x110 = torch.randn(torch.Size([1, 288, 7, 7]))
start = time.time()
output = m(x110)
end = time.time()
print(end-start)
