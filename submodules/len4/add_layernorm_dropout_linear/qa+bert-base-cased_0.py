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
        self.layernorm0 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.dropout0 = Dropout(p=0.1, inplace=False)
        self.linear0 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x23, x25):
        x26=operator.add(x23, x25)
        x27=self.layernorm0(x26)
        x28=self.dropout0(x27)
        x29=self.linear0(x28)
        return x29

m = M().eval()
x23 = torch.randn(torch.Size([1, 384, 768]))
x25 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x23, x25)
end = time.time()
print(end-start)
