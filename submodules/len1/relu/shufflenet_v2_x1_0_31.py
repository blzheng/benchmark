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
        self.relu31 = ReLU(inplace=True)

    def forward(self, x306):
        x307=self.relu31(x306)
        return x307

m = M().eval()
x306 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x306)
end = time.time()
print(end-start)