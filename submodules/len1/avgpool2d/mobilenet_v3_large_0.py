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
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x32):
        x33=self.adaptiveavgpool2d0(x32)
        return x33

m = M().eval()
x32 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x32)
end = time.time()
print(end-start)
