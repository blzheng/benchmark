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
        self.relu75 = ReLU()
        self.conv2d97 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()

    def forward(self, x305, x303):
        x306=self.relu75(x305)
        x307=self.conv2d97(x306)
        x308=self.sigmoid18(x307)
        x309=operator.mul(x308, x303)
        return x309

m = M().eval()
x305 = torch.randn(torch.Size([1, 84, 1, 1]))
x303 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x305, x303)
end = time.time()
print(end-start)
