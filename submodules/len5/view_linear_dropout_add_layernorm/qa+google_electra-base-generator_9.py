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
        self.linear58 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout29 = Dropout(p=0.1, inplace=False)
        self.layernorm19 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x435, x438, x407):
        x439=x435.view(x438)
        x440=self.linear58(x439)
        x441=self.dropout29(x440)
        x442=operator.add(x441, x407)
        x443=self.layernorm19(x442)
        return x443

m = M().eval()
x435 = torch.randn(torch.Size([1, 384, 4, 64]))
x438 = (1, 384, 256, )
x407 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x435, x438, x407)
end = time.time()
print(end-start)