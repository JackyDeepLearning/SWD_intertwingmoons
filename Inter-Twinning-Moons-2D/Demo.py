import torch.nn as nn
import torch.optim

class Generator(nn.Module):

    def __init__(self,Channels):

        super(Generator, self).__init__()
        self.l1   = nn.Linear(       2, Channels)
        self.l2   = nn.Linear(Channels, Channels)
        self.l3   = nn.Linear(Channels, Channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

    def forward(self, x):

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))

        return x

class Classifier1(nn.Module):

    def __init__(self, Channels):

        super(Classifier1, self).__init__()
        self.f1_1 = nn.Linear(Channels, Channels)
        self.f1_2 = nn.Linear(Channels, Channels)
        self.f1_3 = nn.Linear(Channels,        1)
        self.relu = nn.ReLU(inplace=True)
        self.sgmd = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

    def forward(self, x):

        x = self.relu(self.f1_1(x))
        x = self.relu(self.f1_2(x))
        x = self.sgmd(self.f1_3(x))

        return x

class Classifier2(nn.Module):

    def __init__(self, Channels):

        super(Classifier2, self).__init__()
        self.f2_1 = nn.Linear(Channels, Channels)
        self.f2_2 = nn.Linear(Channels, Channels)
        self.f2_3 = nn.Linear(Channels,        1)
        self.relu = nn.ReLU(inplace=True)
        self.sgmd = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.relu(self.f2_1(x))
        x = self.relu(self.f2_2(x))
        x = self.sgmd(self.f2_3(x))

        return x

def AllToyNet(channels, lr):

    g  = Generator(channels).train()
    f1 = Classifier1(channels).train()
    f2 = Classifier2(channels).train()

    step_all = torch.optim.SGD(list(g.parameters())+list(f1.parameters())+list(f2.parameters()), lr)
    step_f1  = torch.optim.SGD(f1.parameters(), lr)
    step_f2  = torch.optim.SGD(f2.parameters(), lr)
    step_gen = torch.optim.SGD(g.parameters(),lr)

    step_f1.zero_grad()
    step_f2.zero_grad()
    step_all.zero_grad()
    step_gen.zero_grad()

    return g, f1, f2, step_f1, step_f2, step_gen, step_all