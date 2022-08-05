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
        self.dropout9 = Dropout(p=0.1, inplace=False)
        self.layernorm6 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x151, x148):
        x152=self.dropout9(x151)
        x153=operator.add(x152, x148)
        x154=self.layernorm6(x153)
        x155=self.linear18(x154)
        return x155

m = M().eval()
x151 = torch.randn(torch.Size([1, 384, 768]))
x148 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x151, x148)
end = time.time()
print(end-start)
