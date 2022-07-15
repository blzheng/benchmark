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
        self.adaptiveavgpool2d1 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x29):
        x30=self.adaptiveavgpool2d1(x29)
        return x30

m = M().eval()
x29 = torch.randn(torch.Size([1, 104, 28, 28]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
