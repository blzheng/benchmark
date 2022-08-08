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

    def forward(self, x463, x436):
        x464=x463.flatten(2)
        x465=self.linear4(x464)
        x466=self.dropout2(x465)
        x467=operator.add(x436, x466)
        x468=self.layernorm1(x467)
        return x468

m = M().eval()
x463 = torch.randn(torch.Size([1, 384, 12, 64]))
x436 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x463, x436)
end = time.time()
print(end-start)
