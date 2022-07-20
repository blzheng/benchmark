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
        self.relu121 = ReLU(inplace=True)

    def forward(self, x418, x410):
        x419=operator.add(x418, x410)
        x420=self.relu121(x419)
        return x420

m = M().eval()
x418 = torch.randn(torch.Size([1, 1024, 14, 14]))
x410 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x418, x410)
end = time.time()
print(end-start)
