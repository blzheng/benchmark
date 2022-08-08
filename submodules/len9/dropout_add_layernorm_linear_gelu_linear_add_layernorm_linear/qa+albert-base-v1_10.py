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
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x428, x399):
        x429=self.dropout2(x428)
        x430=operator.add(x399, x429)
        x431=self.layernorm1(x430)
        x432=self.linear5(x431)
        x433=torch._C._nn.gelu(x432)
        x434=self.linear6(x433)
        x435=operator.add(x434, x431)
        x436=self.layernorm2(x435)
        x437=self.linear1(x436)
        return x437

m = M().eval()
x428 = torch.randn(torch.Size([1, 384, 768]))
x399 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x428, x399)
end = time.time()
print(end-start)