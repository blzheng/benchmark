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
        self.sigmoid10 = Sigmoid()

    def forward(self, x285, x281):
        x286=self.sigmoid10(x285)
        x287=operator.mul(x286, x281)
        return x287

m = M().eval()
x285 = torch.randn(torch.Size([1, 1152, 1, 1]))
x281 = torch.randn(torch.Size([1, 1152, 14, 14]))
start = time.time()
output = m(x285, x281)
end = time.time()
print(end-start)
