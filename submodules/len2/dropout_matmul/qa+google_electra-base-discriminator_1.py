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
        self.dropout4 = Dropout(p=0.1, inplace=False)

    def forward(self, x94, x83):
        x95=self.dropout4(x94)
        x96=torch.matmul(x95, x83)
        return x96

m = M().eval()
x94 = torch.randn(torch.Size([1, 12, 384, 384]))
x83 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x94, x83)
end = time.time()
print(end-start)
