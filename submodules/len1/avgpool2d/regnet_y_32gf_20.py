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
        self.adaptiveavgpool2d20 = AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x331):
        x332=self.adaptiveavgpool2d20(x331)
        return x332

m = M().eval()
x331 = torch.randn(torch.Size([1, 3712, 7, 7]))
start = time.time()
output = m(x331)
end = time.time()
print(end-start)