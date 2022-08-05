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
        self.relu34 = ReLU(inplace=True)

    def forward(self, x130, x122):
        x131=operator.add(x130, x122)
        x132=self.relu34(x131)
        return x132

m = M().eval()
x130 = torch.randn(torch.Size([1, 1024, 28, 28]))
x122 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x130, x122)
end = time.time()
print(end-start)
