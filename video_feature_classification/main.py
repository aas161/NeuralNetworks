import sys
sys.path.append("../")
import os
import pickle
import random
from pathlib import Path
import numpy as np
from accuracy import Accuracy
from tqdm import tqdm
from pytorch.common.datasets_parsers.av_parser import AVDBParser


def get_data(dataset_root, file_list, max_num_clips=0, max_num_samples=50):
    dataset_parser = AVDBParser(
        dataset_root,
        os.path.join(dataset_root, file_list),
        max_num_clips=max_num_clips,
        max_num_samples=max_num_samples,
        ungroup=False,
        load_image=True,
    )
    data = dataset_parser.get_data()
    print("clips count:", len(data))
    print("frames count:", dataset_parser.get_dataset_size())
    return data


def calc_features(data):
    progresser = tqdm(
        iterable=range(0, len(data)),
        desc="calc video features",
        total=len(data),
        unit="files",
    )

    feat, targets = [], []
    for i in progresser:
        clip = data[i]

        rm_list = []
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

            # Угол между подбородком и границами губ
            vec3 = np.array(sample.landmarks[48]) - np.array(sample.landmarks[8])
            vec4 = np.array(sample.landmarks[54]) - np.array(sample.landmarks[8])
            mouth_chin_angle = np.arccos(
                np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

            # Расстояние между вертикальными границами губ
            mouth_distance_vert = np.sqrt((sample.landmarks[51][0] - sample.landmarks[57][0]) ** 2 +
                                     (sample.landmarks[51][1] - sample.landmarks[57][1]) ** 2)

            # Расстояние между горизонтальными границами губ
            mouth_distance_hor = np.sqrt((sample.landmarks[48][0] - sample.landmarks[54][0]) ** 2 +
                                     (sample.landmarks[48][1] - sample.landmarks[54][1]) ** 2)

            # Расстояние между точками на глазах и горизонтальными границами губ (лево)
            eye_mouth_distance_1 = np.sqrt((sample.landmarks[36][0] - sample.landmarks[48][0]) ** 2 +
                                         (sample.landmarks[36][1] - sample.landmarks[48][1]) ** 2)

            # Расстояние между точками на глазах и горизонтальными границами губ (право)
            eye_mouth_distance_2 = np.sqrt((sample.landmarks[45][0] - sample.landmarks[54][0]) ** 2 +
                                         (sample.landmarks[45][1] - sample.landmarks[54][1]) ** 2)

            # Расстояние между точками на бровях и глазах (лево)
            eye_brow_distance_1 = np.sqrt((sample.landmarks[21][0] - sample.landmarks[39][0]) ** 2 +
                                        (sample.landmarks[21][1] - sample.landmarks[39][1]) ** 2)

            # Расстояние между точками на бровях и глазах (право)
            eye_brow_distance_2 = np.sqrt((sample.landmarks[26][0] - sample.landmarks[45][0]) ** 2 +
                                          (sample.landmarks[26][1] - sample.landmarks[45][1]) ** 2)

            feat.append(dist + [mouth_nose_angle, mouth_chin_angle, mouth_distance_vert, mouth_distance_hor,
                                eye_mouth_distance_1, eye_mouth_distance_2, eye_brow_distance_1, eye_brow_distance_2])
            targets.append(sample.labels)

        for sample in rm_list:
            clip.data_samples.remove(sample)

    print("train frames count:", len(feat))
    print("features count:", len(feat[1]))
    print("targets count:", len(targets))

    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim: int = 0):
    if pca_dim > 0:
        pass
        # TODO: выполните сокращение размерности признаков с использованием PCA

    # shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # TODO: используйте классификаторы из sklearn

    y_pred = []
    accuracy_fn.by_frames(y_pred)
    accuracy_fn.by_clips(y_pred)


if __name__ == "__main__":
    experiment_name = "exp_1"
    max_num_clips = 0  # загружайте только часть данных для отладки кода
    use_dump = False  # используйте dump для быстрой загрузки рассчитанных фич из файла

    base_dir = Path(r"C:\Users\zacep\Downloads\NeuralNetworksData\data.part1")
    if 1:
        train_dataset_root = base_dir / "Ryerson/Video"
        train_file_list = base_dir / "Ryerson/train_data_with_landmarks.txt"

        test_dataset_root = base_dir / "Ryerson/Video"
        test_file_list = base_dir / "Ryerson/test_data_with_landmarks.txt"
    else:
        train_dataset_root = base_dir / "OMGEmotionChallenge/omg_TrainVideos/frames"
        train_file_list = base_dir / "OMGEmotionChallenge/omg_TrainVideos/train_data_with_landmarks.txt"

        test_dataset_root = base_dir / "OMGEmotionChallenge/omg_ValidVideos/frames"
        test_file_list = base_dir / "OMGEmotionChallenge/omg_ValidVideos/valid_data_with_landmarks.txt"

    if not use_dump:
        # Загрузка датасета
        train_data = get_data(train_dataset_root, train_file_list, max_num_clips=0)
        test_data = get_data(test_dataset_root, test_file_list, max_num_clips=0)

        # Вычисление признаков
        train_feat, train_targets = calc_features(train_data)
        test_feat, test_targets = calc_features(test_data)

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        # with open(experiment_name + '.pickle', 'wb') as f:
        #    pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        with open(experiment_name + ".pickle", "rb") as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(f)

    # Выполнение классификации
    classification(
        train_feat,
        test_feat,
        train_targets,
        test_targets,
        accuracy_fn=accuracy_fn,
        pca_dim=0,
    )
