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
        self.dropout30 = Dropout(p=0.0, inplace=False)

    def forward(self, x382):
        x383=self.dropout30(x382)
        return x383

m = M().eval()
x382 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x382)
end = time.time()
print(end-start)
