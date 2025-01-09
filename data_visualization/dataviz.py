import sys
sys.path.append("../")
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.spatial.distance import squareform
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from t_sne.tsne import TSNE as torchTSNE
from data_visualization.t_sne.wrapper import Wrapper
from pytorch.common.datasets_parsers.av_parser import AVDBParser


def get_data(dataset_root, file_list, max_num_clips=0):
    dataset_parser = AVDBParser(
        dataset_root,
        os.path.join(dataset_root, file_list),
        max_num_clips=max_num_clips,
        ungroup=False,
        load_image=False,
    )
    data = dataset_parser.get_data()
    print("clips count:", len(data))
    print("total frames count:", dataset_parser.get_dataset_size())
    return data

def calc_features(data, draw: bool = True):
    feat, targets = [], []
    for clip in data:
        for i, sample in enumerate(clip.data_samples):
            if i % 2 != 0:
                continue

            dist = []
            lm_ref = sample.landmarks[30]  # точка на носу
            # Расчет расстояний от точки на носу до всех остальных точек
            for j in range(len(sample.landmarks)):
                lm = sample.landmarks[j]
                dist.append(np.sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))

            # Угол между носом и горизонтальными границами губ
            vec1 = np.array(sample.landmarks[48]) - np.array(sample.landmarks[30])
            vec2 = np.array(sample.landmarks[54]) - np.array(sample.landmarks[30])
            mouth_nose_angle = np.arccos(
                np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

            # Расстояние между вертикальными границами губ
            mouth_distance_vert = np.sqrt((sample.landmarks[51][0] - sample.landmarks[57][0]) ** 2 +
                                     (sample.landmarks[51][1] - sample.landmarks[57][1]) ** 2)

            # Расстояние между горизонтальными границами губ
            mouth_distance_hor = np.sqrt((sample.landmarks[48][0] - sample.landmarks[54][0]) ** 2 +
                                     (sample.landmarks[48][1] - sample.landmarks[54][1]) ** 2)

            # Расстояние между точками на глазах и границами губ
            eye_mouth_distance = np.sqrt((sample.landmarks[36][0] - sample.landmarks[48][0]) ** 2 +
                                         (sample.landmarks[45][1] - sample.landmarks[54][1]) ** 2)

            feat.append(dist + [mouth_nose_angle, mouth_distance_vert, mouth_distance_hor, eye_mouth_distance])
            targets.append(sample.labels)

            if draw:
                img = cv2.imread(sample.img_rel_path)
                for lm in sample.landmarks:
                    cv2.circle(img, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)
                cv2.imshow(sample.text_labels, img)
                cv2.waitKey(100)

    print("train frames count:", len(feat))
    print("features count:", len(feat[1]))
    print("targets count:", len(targets))

    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)

def run_tsne(feat, targets, pca_dim=50, tsne_dim=2):
    # Применение PCA
    if pca_dim > 0:
        feat = PCA(n_components=pca_dim).fit_transform(feat)

    # Вычисление попарных расстояний
    distances2 = pairwise_distances(feat, metric="euclidean", squared=True)

    # Вычисление P-распределения
    pij = manifold._t_sne._joint_probabilities(distances2, 100, False)
    pij = squareform(pij)

    # Подготовка индексов и вероятностей
    i, j = np.indices(pij.shape)
    i, j = i.ravel(), j.ravel()
    pij = pij.ravel().astype("float32")

    # Удаление диагональных элементов
    idx = i != j
    i, j, pij = i[idx], j[idx], pij[idx]

    # Нормализация P-распределения
    pij = pij / np.sum(pij)
    pij = torch.from_numpy(pij)

    # Обучение модели
    epochs = 250
    model = torchTSNE(n_points=feat.shape[0], n_dim=tsne_dim)
    w = Wrapper(model, batchsize=feat.shape[0], targets=targets, epochs=epochs)
    w.fit(pij, i, j)
    w.plot_losses(epochs)

if __name__ == "__main__":
    # Путь до датасета
    base_dir = Path(r"C:\Users\zacep\Downloads\NeuralNetworksData\data.part1")
    if 1:
        train_dataset_root = base_dir / "Ryerson/Video"
        train_file_list = base_dir / "Ryerson/train_data_with_landmarks.txt"
    elif 0:
        train_dataset_root = base_dir / "AFEW-VA/crop"
        train_file_list = base_dir / "AFEW-VA/train_data_with_landmarks.txt"
    elif 0:
        train_dataset_root = base_dir / "OMGEmotionChallenge/omg_TrainVideos/frames"
        train_file_list = base_dir / "OMGEmotionChallenge/omg_TrainVideos/train_data_with_landmarks.txt"

    # Загрузка данных
    data = get_data(train_dataset_root, train_file_list, max_num_clips=0)

    # Получение признаков и меток
    feat, targets = calc_features(data, draw=False)

    # Запуск t-SNE
    run_tsne(feat, targets, pca_dim=0)
