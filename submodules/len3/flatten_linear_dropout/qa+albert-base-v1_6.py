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
        self.linear4 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)

    def forward(self, x278):
        x279=x278.flatten(2)
        x280=self.linear4(x279)
        x281=self.dropout2(x280)
        return x281

m = M().eval()
x278 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x278)
end = time.time()
print(end-start)
