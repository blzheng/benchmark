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
        self.dropout10 = Dropout(p=0.1, inplace=False)

    def forward(self, x178, x167):
        x179=self.dropout10(x178)
        x180=torch.matmul(x179, x167)
        return x180

m = M().eval()
x178 = torch.randn(torch.Size([1, 12, 384, 384]))
x167 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x178, x167)
end = time.time()
print(end-start)
