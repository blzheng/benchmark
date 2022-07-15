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

    def forward(self, x89):
        x90=self.adaptiveavgpool2d6(x89)
        return x90

m = M().eval()
x89 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x89)
end = time.time()
print(end-start)
