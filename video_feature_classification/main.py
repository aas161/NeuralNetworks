import sys
sys.path.append("../")
import os
import pickle
import random
from pathlib import Path
import numpy as np
from math import sqrt
from accuracy import Accuracy
from tqdm import tqdm
from pytorch.common.datasets_parsers.av_parser import AVDBParser
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
    for clip_idx in progresser:
        clip = data[clip_idx]

        # Сохраняем признаки предыдущего кадра, чтобы вычислить дельту
        prev_vector = None

        for i, sample in enumerate(clip.data_samples):
            dist = []
            lm_ref = sample.landmarks[30]  # точка на носу
            # Расчет расстояний от точки на носу до всех остальных точек
            for j in range(len(sample.landmarks)):
                lm = sample.landmarks[j]
                dist.append(sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))

            # Угол между носом и горизонтальными границами губ
            vec1 = np.array(sample.landmarks[48]) - np.array(sample.landmarks[30])
            vec2 = np.array(sample.landmarks[54]) - np.array(sample.landmarks[30])
            mouth_nose_angle = np.arccos(
                np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)
            )

            # Угол между подбородком и границами губ
            vec3 = np.array(sample.landmarks[48]) - np.array(sample.landmarks[8])
            vec4 = np.array(sample.landmarks[54]) - np.array(sample.landmarks[8])
            mouth_chin_angle = np.arccos(
                np.clip(np.dot(vec3, vec4) / (np.linalg.norm(vec3) * np.linalg.norm(vec4)), -1.0, 1.0)
            )

            # Расстояние между вертикальными границами губ
            mouth_distance_vert = sqrt((sample.landmarks[51][0] - sample.landmarks[57][0]) ** 2 +
                                       (sample.landmarks[51][1] - sample.landmarks[57][1]) ** 2)

            # Расстояние между горизонтальными границами губ
            mouth_distance_hor = sqrt((sample.landmarks[48][0] - sample.landmarks[54][0]) ** 2 +
                                      (sample.landmarks[48][1] - sample.landmarks[54][1]) ** 2)

            # Расстояние между точками на глазах и горизонтальными границами губ
            eye_mouth_distance_l = sqrt((sample.landmarks[36][0] - sample.landmarks[48][0]) ** 2 +
                                        (sample.landmarks[36][1] - sample.landmarks[48][1]) ** 2)
            eye_mouth_distance_r = sqrt((sample.landmarks[45][0] - sample.landmarks[54][0]) ** 2 +
                                        (sample.landmarks[45][1] - sample.landmarks[54][1]) ** 2)

            # Расстояние между точками на бровях и глазах
            eye_brow_distance_l = sqrt((sample.landmarks[19][0] - sample.landmarks[37][0]) ** 2 +
                                       (sample.landmarks[19][1] - sample.landmarks[37][1]) ** 2)
            eye_brow_distance_r = sqrt((sample.landmarks[24][0] - sample.landmarks[44][0]) ** 2 +
                                       (sample.landmarks[24][1] - sample.landmarks[44][1]) ** 2)

            # Собираем текущий вектор признаков
            current_features = [
                mouth_nose_angle,
                mouth_chin_angle,
                mouth_distance_vert,
                mouth_distance_hor,
                eye_mouth_distance_l,
                eye_mouth_distance_r,
                eye_brow_distance_l,
                eye_brow_distance_r
            ] + dist

            # Вычисляем дельту между текущим и предыдущим кадром (если prev_vector доступен)
            if prev_vector is not None:
                delta_features = [cur - prev for cur, prev in zip(current_features, prev_vector)]
            else:
                # Если это первый кадр в клипе, дельты нет, ставим нули
                delta_features = [0.0] * len(current_features)

            # Объединяем оба набора признаков: базовые + временные
            combined_features = current_features + delta_features

            feat.append(combined_features)
            targets.append(sample.labels)

            # Обновляем prev_vector
            prev_vector = current_features

    print("Total frames processed:", len(feat))
    print("Feature dimension per frame:", len(feat[0]))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)

def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim: int = 0):
    # Перемешивание данных
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # Стандартизация признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Сокращение размерности признаков
    if pca_dim > 0:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Сетка гиперпараметров для RandomForest
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # GridSearchCV для подбора параметров
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation score (F1):", grid_search.best_score_)

    # Итоговый классификатор с лучшими параметрами
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    accuracy_fn.by_frames(y_pred)
    accuracy_fn.by_clips(y_pred)

if __name__ == "__main__":

    experiment_name = "exp_09_01_2025"
    max_num_clips = 0
    use_dump = True  # dump для быстрой загрузки из файла

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

        # Создание объекта для расчета метрик
        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        # Сохранение рассчитанных признаков на диск
        with open(experiment_name + '.pickle', 'wb') as f:
           pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        # Загрузка рассчитанных признаков с диска
        with open(experiment_name + ".pickle", "rb") as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(f)

    # Выполнение классификации
    classification(
        train_feat,
        test_feat,
        train_targets,
        test_targets,
        accuracy_fn=accuracy_fn,
        pca_dim=50,
    )
