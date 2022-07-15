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

    def forward(self, x359):
        x360=self.adaptiveavgpool2d23(x359)
        return x360

m = M().eval()
x359 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x359)
end = time.time()
print(end-start)
