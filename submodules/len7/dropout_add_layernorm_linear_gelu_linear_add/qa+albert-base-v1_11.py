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
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x465, x436):
        x466=self.dropout2(x465)
        x467=operator.add(x436, x466)
        x468=self.layernorm1(x467)
        x469=self.linear5(x468)
        x470=torch._C._nn.gelu(x469)
        x471=self.linear6(x470)
        x472=operator.add(x471, x468)
        return x472

m = M().eval()
x465 = torch.randn(torch.Size([1, 384, 768]))
x436 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x465, x436)
end = time.time()
print(end-start)
