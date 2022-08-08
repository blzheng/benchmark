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
        self.linear65 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout33 = Dropout(p=0.1, inplace=False)

    def forward(self, x485):
        x486=torch._C._nn.gelu(x485)
        x487=self.linear65(x486)
        x488=self.dropout33(x487)
        return x488

m = M().eval()
x485 = torch.randn(torch.Size([1, 384, 3072]))
start = time.time()
output = m(x485)
end = time.time()
print(end-start)