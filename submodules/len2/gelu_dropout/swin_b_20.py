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
        self.gelu20 = GELU(approximate='none')
        self.dropout40 = Dropout(p=0.0, inplace=False)

    def forward(self, x496):
        x497=self.gelu20(x496)
        x498=self.dropout40(x497)
        return x498

m = M().eval()
x496 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x496)
end = time.time()
print(end-start)
