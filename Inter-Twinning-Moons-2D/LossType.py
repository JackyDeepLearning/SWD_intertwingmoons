import torch
import torch.nn as nn

def loss_type(mode,lp1,lp2,l):

    if mode == 0:
        bce  = nn.BCELoss()
        loss = bce(lp1,l) +bce(lp2,l)
    elif mode == 1:
        loss = nn.CrossEntropyLoss()
    elif mode == 2:
        mse1 = MSELoss(lp1,l)
        mse2 = MSELoss(lp2,l)
        loss = mse1 + mse2
    elif mode == 3:
        loss = MCDLoss(lp1,lp2)
    elif mode == 4:
        loss = SWDLoss(lp1,lp2)
    elif mode == 5:
        loss = CostLoss(lp1,lp2,l)

    return loss


def MSELoss(lp,l):

    mseloss = sum([(y - x) ^ 2 for x,y in zip(lp,l)]) / len(l)

    return mseloss

def MCDLoss(p1,p2):

    mcdloss = torch.mean(torch.abs(p1 - p2))

    return mcdloss

def SWDLoss(p1,p2):

    PShape = p1.shape

    if PShape[1] > 1:

        Projection = torch.randn(PShape[1], 128)
        Projection = torch.rsqrt(torch.smm(torch.mul(Projection,Projection), 0, keepdim = True)) * Projection

        p1 = torch.matmul(p1,Projection)
        p2 = torch.matmul(p2,Projection)

    p1  = torch.topk(p1, PShape[0], dim=0)[0]
    p2  = torch.topk(p2, PShape[0], dim=0)[0]
    DIS = p1 - p2
    SWD = torch.mean(torch.mul(DIS, DIS))

    return SWD

def CostLoss(lp1,lp2,l):

    eps   = 1e-5
    cost1 = torch.sum((-1)*(l * torch.log(lp1 + eps) + (1 - l) * torch.log(1 - lp1 + eps))) / len(l)
    cost2 = torch.sum((-1)*(l * torch.log(lp2 + eps) + (1 - l) * torch.log(1 - lp2 + eps))) / len(l)
    loss  = cost1 + cost2

    return loss