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
        self.adaptiveavgpool2d3 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x46):
        x47=self.adaptiveavgpool2d3(x46)
        return x47

m = M().eval()
x46 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x46)
end = time.time()
print(end-start)
