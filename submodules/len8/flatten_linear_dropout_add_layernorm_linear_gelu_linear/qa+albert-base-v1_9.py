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
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x389, x362):
        x390=x389.flatten(2)
        x391=self.linear4(x390)
        x392=self.dropout2(x391)
        x393=operator.add(x362, x392)
        x394=self.layernorm1(x393)
        x395=self.linear5(x394)
        x396=torch._C._nn.gelu(x395)
        x397=self.linear6(x396)
        return x397

m = M().eval()
x389 = torch.randn(torch.Size([1, 384, 12, 64]))
x362 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x389, x362)
end = time.time()
print(end-start)
