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
        self.layernorm30 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear28 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu13 = GELU(approximate='none')
        self.dropout26 = Dropout(p=0.0, inplace=False)
        self.linear29 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout27 = Dropout(p=0.0, inplace=False)

    def forward(self, x318, x332):
        x333=operator.add(x318, x332)
        x334=self.layernorm30(x333)
        x335=self.linear28(x334)
        x336=self.gelu13(x335)
        x337=self.dropout26(x336)
        x338=self.linear29(x337)
        x339=self.dropout27(x338)
        return x339

m = M().eval()
x318 = torch.randn(torch.Size([1, 14, 14, 384]))
x332 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x318, x332)
end = time.time()
print(end-start)
