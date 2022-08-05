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
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x206):
        x207=self.dropout1(x206)
        return x207

m = M().eval()
x206 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x206)
end = time.time()
print(end-start)
