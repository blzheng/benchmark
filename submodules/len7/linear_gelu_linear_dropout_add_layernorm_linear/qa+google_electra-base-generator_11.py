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
        self.linear71 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear72 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout36 = Dropout(p=0.1, inplace=False)
        self.layernorm24 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear73 = Linear(in_features=256, out_features=2, bias=True)

    def forward(self, x527, x527):
        x528=self.linear71(x527)
        x529=torch._C._nn.gelu(x528)
        x530=self.linear72(x529)
        x531=self.dropout36(x530)
        x532=operator.add(x531, x527)
        x533=self.layernorm24(x532)
        x534=self.linear73(x533)
        return x534

m = M().eval()
x527 = torch.randn(torch.Size([1, 384, 256]))
x527 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x527, x527)
end = time.time()
print(end-start)
