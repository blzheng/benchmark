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

    def forward(self, x580):
        x581=self.adaptiveavgpool2d31(x580)
        return x581

m = M().eval()
x580 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x580)
end = time.time()
print(end-start)
