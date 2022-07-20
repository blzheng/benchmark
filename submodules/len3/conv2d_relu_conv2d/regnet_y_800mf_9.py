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
        self.conv2d51 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu39 = ReLU()
        self.conv2d52 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x160):
        x161=self.conv2d51(x160)
        x162=self.relu39(x161)
        x163=self.conv2d52(x162)
        return x163

m = M().eval()
x160 = torch.randn(torch.Size([1, 320, 1, 1]))
start = time.time()
output = m(x160)
end = time.time()
print(end-start)
