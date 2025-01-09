from torch import nn
import torch


class TSNE(nn.Module):
    def __init__(self, n_points, n_dim):
        super().__init__()
        self.n_points = n_points
        self.n_dim = n_dim

        # Инициализация эмбеддингов с малыми случайными значениями
        self.logits = nn.Embedding(n_points, n_dim)
        self.logits.weight.data.normal_(0, 1e-4)

    def forward(self, pij, i, j):
        # Получаем эмбеддинги
        z = self.logits.weight
        z_i = z[i]
        z_j = z[j]

        # Вычисляем попарные расстояния
        distances = torch.sum((z_i - z_j) ** 2, dim=1)

        # Вычисляем Q-распределение (t-распределение Стьюдента)
        eps = 1e-12
        q = 1. / (1. + distances)

        # Нормализация Q-распределения
        q = q / torch.sum(q)

        # Вычисляем KL-дивергенцию
        # KL(P||Q) = Σ p_i * (log(p_i) - log(q_i))
        loss = torch.sum(pij * (torch.log(pij + eps) - torch.log(q + eps)))

        return loss

    def __call__(self, *args):
        return self.forward(*args)
