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
        self.linear33 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout17 = Dropout(p=0.1, inplace=False)

    def forward(self, x270):
        x271=self.linear33(x270)
        x272=self.dropout17(x271)
        return x272

m = M().eval()
x270 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x270)
end = time.time()
print(end-start)
