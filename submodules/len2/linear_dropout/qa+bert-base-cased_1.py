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
        self.linear5 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout3 = Dropout(p=0.1, inplace=False)

    def forward(self, x66):
        x67=self.linear5(x66)
        x68=self.dropout3(x67)
        return x68

m = M().eval()
x66 = torch.randn(torch.Size([1, 384, 3072]))
start = time.time()
output = m(x66)
end = time.time()
print(end-start)