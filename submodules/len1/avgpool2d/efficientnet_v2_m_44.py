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
        self.adaptiveavgpool2d44 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x783):
        x784=self.adaptiveavgpool2d44(x783)
        return x784

m = M().eval()
x783 = torch.randn(torch.Size([1, 1280, 7, 7]))
start = time.time()
output = m(x783)
end = time.time()
print(end-start)
