import random
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

def chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    endpoints = []
    start = 0
    for stop in range(0, len(args[0]), n):
        if stop - start > 0:
            endpoints.append((start, stop))
            start = stop
    random.shuffle(endpoints)
    for start, stop in endpoints:
        yield [a[start:stop] for a in args]

class Wrapper:
    def __init__(self, model, cuda=True, epochs=5, batchsize=1024):
        self.batchsize = batchsize
        self.epochs = epochs
        self.cuda = cuda
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Оптимизатор

        if cuda:
            self.model.cuda()

        self.losses = []  # Список для хранения потерь

    def fit(self, *args):
        self.model.train()
        if self.cuda:
            self.model.cuda()
        for epoch in range(self.epochs):
            total = 0.0
            for itr, datas in enumerate(chunks(self.batchsize, *args)):
                datas = [Variable(torch.from_numpy(data)) for data in datas]
                if self.cuda:
                    datas = [data.cuda() for data in datas]

                pij, i, j = datas  # Разделяем входные данные
                self.optimizer.zero_grad()  # Обнуляем градиенты

                # Вычисляем логиты и градиенты
                loss = self.model(pij, i, j)  # Вызываем модель
                loss.backward()  # Обратное распространение ошибки
                self.optimizer.step()  # Обновление весов

                total += loss.item()  # Суммируем потери

            # Сохраняем среднее значение потерь за эпоху
            avg_loss = total / (len(args[0]) * 1.0)
            self.losses.append(avg_loss)

            msg = "Train Epoch: {} \tLoss: {:.6e}"
            msg = msg.format(epoch + 1, avg_loss)
            print(msg)

        # Построение графика потерь
        plt.plot(range(1, self.epochs + 1), self.losses)
        plt.xlabel('Epochs')
        plt.ylabel('Kullback–Leibler Loss')
        plt.title('График изменения потерь')
        plt.grid()
        plt.show()