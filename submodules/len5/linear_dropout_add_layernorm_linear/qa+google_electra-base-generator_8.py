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
        self.linear28 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout14 = Dropout(p=0.1, inplace=False)
        self.layernorm9 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear29 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x229, x197):
        x230=self.linear28(x229)
        x231=self.dropout14(x230)
        x232=operator.add(x231, x197)
        x233=self.layernorm9(x232)
        x234=self.linear29(x233)
        return x234

m = M().eval()
x229 = torch.randn(torch.Size([1, 384, 256]))
x197 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x229, x197)
end = time.time()
print(end-start)
