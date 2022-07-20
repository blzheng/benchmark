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
        self.relu77 = ReLU(inplace=True)
        self.conv2d101 = Conv2d(3712, 3712, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
        self.batchnorm2d63 = BatchNorm2d(3712, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu78 = ReLU(inplace=True)

    def forward(self, x317):
        x318=self.relu77(x317)
        x319=self.conv2d101(x318)
        x320=self.batchnorm2d63(x319)
        x321=self.relu78(x320)
        return x321

m = M().eval()
x317 = torch.randn(torch.Size([1, 3712, 14, 14]))
start = time.time()
output = m(x317)
end = time.time()
print(end-start)
