import torch
import torch.nn as nn

class SSL_EntropyLoss(nn.Module):
    def __init__(self, sup_ratio=100):
        super(SSL_EntropyLoss, self).__init__()
        self.sup_ratio = sup_ratio

    def forward(self, predictions, target):
        assert len(predictions.shape) == 2
        assert predictions.shape[0] == target.shape[0]
        # unsupervised part (entropy)
        loss_uns = 0 if (target == -1).sum() == 0 else -torch.log(predictions[target == -1]).sum(dim=1).mean()
        # supervised part (cross entropy)
        loss_ce = nn.CrossEntropyLoss()
        loss_sup = 0 if (target != -1).sum() == 0 else loss_ce(predictions[target != -1],
                                                               target[target != -1].long()).mean()
        loss = loss_uns + self.sup_ratio * loss_sup
        return loss


def test(sup_ratio=100):
    target = torch.ones(13)
    target[3] = -1
    sm = nn.Softmax(dim=1)
    predictions = sm(torch.randn((13, 10)))
    print(target.type(), predictions.type())
    print(target.shape, predictions.shape)
    loss_func = SSL_EntropyLoss(sup_ratio=sup_ratio)
    loss = loss_func(predictions, target)
    print(loss)


if __name__ == '__main__':
    test()
