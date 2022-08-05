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
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x94, x66):
        x95=self.linear4(x94)
        x96=self.dropout2(x95)
        x97=operator.add(x66, x96)
        x98=self.layernorm1(x97)
        return x98

m = M().eval()
x94 = torch.randn(torch.Size([1, 384, 768]))
x66 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x94, x66)
end = time.time()
print(end-start)
