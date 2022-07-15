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
        self.adaptiveavgpool2d14 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x239):
        x240=self.adaptiveavgpool2d14(x239)
        return x240

m = M().eval()
x239 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x239)
end = time.time()
print(end-start)
