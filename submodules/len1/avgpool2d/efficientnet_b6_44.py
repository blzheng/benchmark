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
        self.adaptiveavgpool2d44 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x690):
        x691=self.adaptiveavgpool2d44(x690)
        return x691

m = M().eval()
x690 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x690)
end = time.time()
print(end-start)
