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

    def forward(self, x94, x84):
        x95=torch.nn.functional.softmax(x94,dim=-1, _stacklevel=3, dtype=None)
        x96=self.dropout4(x95)
        x97=torch.matmul(x96, x84)
        return x97

m = M().eval()
x94 = torch.randn(torch.Size([1, 4, 384, 384]))
x84 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x94, x84)
end = time.time()
print(end-start)
