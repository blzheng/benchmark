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
        self.adaptiveavgpool2d11 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x170):
        x171=self.adaptiveavgpool2d11(x170)
        return x171

m = M().eval()
x170 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x170)
end = time.time()
print(end-start)
