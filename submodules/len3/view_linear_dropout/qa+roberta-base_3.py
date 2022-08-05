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
        self.linear21 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout11 = Dropout(p=0.1, inplace=False)

    def forward(self, x182, x185):
        x186=x182.view(x185)
        x187=self.linear21(x186)
        x188=self.dropout11(x187)
        return x188

m = M().eval()
x182 = torch.randn(torch.Size([1, 384, 12, 64]))
x185 = (1, 384, 768, )
start = time.time()
output = m(x182, x185)
end = time.time()
print(end-start)
