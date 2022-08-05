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
        self.linear22 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout11 = Dropout(p=0.1, inplace=False)

    def forward(self, x183, x186):
        x187=x183.view(x186)
        x188=self.linear22(x187)
        x189=self.dropout11(x188)
        return x189

m = M().eval()
x183 = torch.randn(torch.Size([1, 384, 4, 64]))
x186 = (1, 384, 256, )
start = time.time()
output = m(x183, x186)
end = time.time()
print(end-start)
