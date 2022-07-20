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

    def forward(self, x206):
        x207=self.adaptiveavgpool2d0(x206)
        return x207

m = M().eval()
x206 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x206)
end = time.time()
print(end-start)