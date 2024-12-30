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

        # Вычисляем евклидово расстояние между точками
        distances = F.pairwise_distance(z_i, z_j) ** 2  # Квадрат расстояния

        # Вычисляем матрицу сходства
        qij = 1 / (1 + distances)  # t-SNE формула для q_ij
        qij = qij / qij.sum()  # Нормируем

        # Вычисляем потери Кульбака-Лейблера
        loss_kld = F.kl_div(torch.log(qij + 1e-10), pij, reduction='sum')  # Добавляем маленькое значение для предотвращения логарифмирования нуля

        return loss_kld

    def __call__(self, *args):
        return self.forward(*args)
