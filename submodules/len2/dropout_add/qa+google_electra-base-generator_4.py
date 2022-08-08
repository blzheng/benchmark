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
        self.dropout8 = Dropout(p=0.1, inplace=False)

    def forward(self, x146, x113):
        x147=self.dropout8(x146)
        x148=operator.add(x147, x113)
        return x148

m = M().eval()
x146 = torch.randn(torch.Size([1, 384, 256]))
x113 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x146, x113)
end = time.time()
print(end-start)
