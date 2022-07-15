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
        self.adaptiveavgpool2d52 = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x815):
        x816=self.adaptiveavgpool2d52(x815)
        return x816

m = M().eval()
x815 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x815)
end = time.time()
print(end-start)
