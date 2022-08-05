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
        self.linear42 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu20 = GELU(approximate='none')
        self.dropout40 = Dropout(p=0.0, inplace=False)

    def forward(self, x495):
        x496=self.linear42(x495)
        x497=self.gelu20(x496)
        x498=self.dropout40(x497)
        return x498

m = M().eval()
x495 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x495)
end = time.time()
print(end-start)
