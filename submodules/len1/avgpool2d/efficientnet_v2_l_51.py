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
        self.adaptiveavgpool2d51 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x933):
        x934=self.adaptiveavgpool2d51(x933)
        return x934

m = M().eval()
x933 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x933)
end = time.time()
print(end-start)
