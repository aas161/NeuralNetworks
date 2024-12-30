from torch import nn
import torch
import torch.nn.functional as F


class TSNE(nn.Module):
    def __init__(self, n_points, n_dim):
        super().__init__()
        self.n_points = n_points
        self.n_dim = n_dim
        self.logits = nn.Embedding(n_points, n_dim)

    def forward(self, pij, i, j):
        # Получаем логиты
        z = self.logits.weight
        z_i = z[i]
        z_j = z[j]

        # Вычисляем евклидово расстояние между всеми точками
        distances = F.pairwise_distance(z_i, z_j) ** 2

        # Совместные вероятности в пространстве отображения
        qij = 1 / (1 + distances)
        qij = qij / qij.sum()

        # Вычисляем расстояние Кульбака-Лейблера
        loss_kld = F.kl_div(torch.log(qij + 1e-10), pij, reduction='sum')

        return loss_kld

    def __call__(self, *args):
        return self.forward(*args)
