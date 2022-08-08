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
        self.linear34 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout17 = Dropout(p=0.1, inplace=False)
        self.layernorm11 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear35 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear36 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout18 = Dropout(p=0.1, inplace=False)

    def forward(self, x267, x270, x239):
        x271=x267.view(x270)
        x272=self.linear34(x271)
        x273=self.dropout17(x272)
        x274=operator.add(x273, x239)
        x275=self.layernorm11(x274)
        x276=self.linear35(x275)
        x277=torch._C._nn.gelu(x276)
        x278=self.linear36(x277)
        x279=self.dropout18(x278)
        x280=operator.add(x279, x275)
        return x280

m = M().eval()
x267 = torch.randn(torch.Size([1, 384, 4, 64]))
x270 = (1, 384, 256, )
x239 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x267, x270, x239)
end = time.time()
print(end-start)
