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
        self.linear69 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout35 = Dropout(p=0.1, inplace=False)
        self.layernorm23 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear70 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear71 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x518, x521, x490):
        x522=x518.view(x521)
        x523=self.linear69(x522)
        x524=self.dropout35(x523)
        x525=operator.add(x524, x490)
        x526=self.layernorm23(x525)
        x527=self.linear70(x526)
        x528=torch._C._nn.gelu(x527)
        x529=self.linear71(x528)
        return x529

m = M().eval()
x518 = torch.randn(torch.Size([1, 384, 12, 64]))
x521 = (1, 384, 768, )
x490 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x518, x521, x490)
end = time.time()
print(end-start)
