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
        self.adaptiveavgpool2d31 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x483):
        x484=self.adaptiveavgpool2d31(x483)
        return x484

m = M().eval()
x483 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x483)
end = time.time()
print(end-start)