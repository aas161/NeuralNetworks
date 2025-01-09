import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time

class Wrapper:
    def __init__(self, model, batchsize, targets, epochs):
        self.batchsize = batchsize
        self.epochs = epochs
        self.targets = targets
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=5)
        self.losses = []
        self.best_loss = float('inf')
        self.no_improvement_epochs = 0
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5)

    def fit(self, pij, i, j):
        self.model.train()
        time_start = time.time()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # Вычисление потерь и обновление весов
            loss = self.model(pij, i, j)
            loss.backward()

            # Ограничение градиентов для стабильности обучения
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

            self.optimizer.step()
            self.losses.append(loss.item())

            print(f"Train Epoch: {epoch + 1} \tLoss: {loss.item():.6e} \tLearning Rate: {self.optimizer.param_groups[0]['lr']:.6e}")

            # Визуализация распределения в двумерном пространстве
            if epoch % 5 == 0:
                embed = self.model.logits.weight.cpu().data.numpy()
                self.draw(embed, self.targets)

            # Проверка улучшения потерь
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.no_improvement_epochs = 0
            else:
                self.no_improvement_epochs += 1

            # Уменьшение скорости обучения
            self.lr_scheduler.step(loss)

            # Ранняя остановка после 10 эпох без улучшений потерь
            if self.no_improvement_epochs >= 10:
                print("Early stopping triggered")
                break

        time_end = time.time()
        print(f"Training time: {time_end - time_start:.2f} s")

    def plot_losses(self):
        # Построение графика потерь
        plt.plot(range(1, self.epochs + 1), self.losses)
        plt.xlabel('Epochs')
        plt.ylabel('Kullback–Leibler Loss')
        plt.grid()
        plt.show()

    def draw(self, points2D, targets, save=False):
        # Визуализация точек
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(points2D[:, 0], points2D[:, 1], c=targets, cmap='Set1', s=40)
        legend = ax.legend(*scatter.legend_elements(), loc="upper right")
        ax.add_artist(legend)
        if save:
            plt.savefig("scatter.png", bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
            plt.pause(5)
            plt.close(fig)