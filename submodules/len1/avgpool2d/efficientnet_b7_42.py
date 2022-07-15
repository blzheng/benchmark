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
        self.adaptiveavgpool2d42 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x657):
        x658=self.adaptiveavgpool2d42(x657)
        return x658

m = M().eval()
x657 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x657)
end = time.time()
print(end-start)
