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

    def forward(self, x146):
        x147=self.dropout8(x146)
        return x147

m = M().eval()
x146 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x146)
end = time.time()
print(end-start)
