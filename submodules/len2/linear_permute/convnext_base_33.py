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
        self.linear67 = Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, x393):
        x394=self.linear67(x393)
        x395=torch.permute(x394, [0, 3, 1, 2])
        return x395

m = M().eval()
x393 = torch.randn(torch.Size([1, 7, 7, 4096]))
start = time.time()
output = m(x393)
end = time.time()
print(end-start)
