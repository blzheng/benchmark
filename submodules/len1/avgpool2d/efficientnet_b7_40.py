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
        self.adaptiveavgpool2d40 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x625):
        x626=self.adaptiveavgpool2d40(x625)
        return x626

m = M().eval()
x625 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x625)
end = time.time()
print(end-start)
