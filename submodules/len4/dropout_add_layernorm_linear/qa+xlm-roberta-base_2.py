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
        self.dropout3 = Dropout(p=0.1, inplace=False)
        self.layernorm2 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear6 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x67, x64):
        x68=self.dropout3(x67)
        x69=operator.add(x68, x64)
        x70=self.layernorm2(x69)
        x71=self.linear6(x70)
        return x71

m = M().eval()
x67 = torch.randn(torch.Size([1, 384, 768]))
x64 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x67, x64)
end = time.time()
print(end-start)
