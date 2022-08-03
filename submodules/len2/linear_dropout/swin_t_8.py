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
        self.linear19 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout17 = Dropout(p=0.0, inplace=False)

    def forward(self, x222):
        x223=self.linear19(x222)
        x224=self.dropout17(x223)
        return x224

m = M().eval()
x222 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x222)
end = time.time()
print(end-start)
