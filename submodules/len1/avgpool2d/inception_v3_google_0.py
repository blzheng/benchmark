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
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x323):
        x324=self.adaptiveavgpool2d0(x323)
        return x324

m = M().eval()
x323 = torch.randn(torch.Size([1, 2048, 5, 5]))
start = time.time()
output = m(x323)
end = time.time()
print(end-start)
