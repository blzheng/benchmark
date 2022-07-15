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
        self.adaptiveavgpool2d7 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x105):
        x106=self.adaptiveavgpool2d7(x105)
        return x106

m = M().eval()
x105 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x105)
end = time.time()
print(end-start)
