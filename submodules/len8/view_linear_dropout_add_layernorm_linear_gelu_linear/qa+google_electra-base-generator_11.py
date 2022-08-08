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
        self.linear70 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout35 = Dropout(p=0.1, inplace=False)
        self.layernorm23 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear71 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear72 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x519, x522, x491):
        x523=x519.view(x522)
        x524=self.linear70(x523)
        x525=self.dropout35(x524)
        x526=operator.add(x525, x491)
        x527=self.layernorm23(x526)
        x528=self.linear71(x527)
        x529=torch._C._nn.gelu(x528)
        x530=self.linear72(x529)
        return x530

m = M().eval()
x519 = torch.randn(torch.Size([1, 384, 4, 64]))
x522 = (1, 384, 256, )
x491 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x519, x522, x491)
end = time.time()
print(end-start)