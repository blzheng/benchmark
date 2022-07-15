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
        self.adaptiveavgpool2d5 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x76):
        x77=self.adaptiveavgpool2d5(x76)
        return x77

m = M().eval()
x76 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x76)
end = time.time()
print(end-start)
