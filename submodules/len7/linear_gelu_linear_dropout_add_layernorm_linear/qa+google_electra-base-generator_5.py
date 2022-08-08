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
        self.linear35 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear36 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout18 = Dropout(p=0.1, inplace=False)
        self.layernorm12 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear37 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x275, x275):
        x276=self.linear35(x275)
        x277=torch._C._nn.gelu(x276)
        x278=self.linear36(x277)
        x279=self.dropout18(x278)
        x280=operator.add(x279, x275)
        x281=self.layernorm12(x280)
        x282=self.linear37(x281)
        return x282

m = M().eval()
x275 = torch.randn(torch.Size([1, 384, 256]))
x275 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x275, x275)
end = time.time()
print(end-start)
