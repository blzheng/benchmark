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
        self.relu66 = ReLU(inplace=True)

    def forward(self, x270):
        x271=self.relu66(x270)
        return x271

m = M().eval()
x270 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x270)
end = time.time()
print(end-start)
