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
        self.adaptiveavgpool2d48 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x885):
        x886=self.adaptiveavgpool2d48(x885)
        return x886

m = M().eval()
x885 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x885)
end = time.time()
print(end-start)
