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
        self.adaptiveavgpool2d29 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x454):
        x455=self.adaptiveavgpool2d29(x454)
        return x455

m = M().eval()
x454 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x454)
end = time.time()
print(end-start)
