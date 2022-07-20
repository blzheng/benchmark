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
        self.relu54 = ReLU(inplace=True)

    def forward(self, x181, x189):
        x190=operator.add(x181, x189)
        x191=self.relu54(x190)
        return x191

m = M().eval()
x181 = torch.randn(torch.Size([1, 400, 7, 7]))
x189 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x181, x189)
end = time.time()
print(end-start)
