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
        self.adaptiveavgpool2d12 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x207):
        x208=self.adaptiveavgpool2d12(x207)
        return x208

m = M().eval()
x207 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x207)
end = time.time()
print(end-start)
