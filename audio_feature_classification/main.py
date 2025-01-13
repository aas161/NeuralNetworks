import sys
sys.path.append("../")
import pickle
import random
from pathlib import Path
import numpy as np
from accuracy import Accuracy
from tqdm import tqdm
from accuracy import AccuracyRegression
from voice_feature_extraction import OpenSMILE
from pytorch.common.datasets_parsers.av_parser import AVDBParser
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def get_data(dataset_root, file_list, max_num_clips: int = 0):
    """Получение данных из датасета"""

    dataset_parser = AVDBParser(dataset_root, file_list, max_num_clips=max_num_clips)
    data = dataset_parser.get_data()
    print("clips count:", len(data))
    print("frames count:", dataset_parser.get_dataset_size())
    return data


def calc_features(data, opensmile_root_dir, opensmile_config_path):
    """"Вычисление признаков с помощью библиотеки OpenSmile"""

    # Инициализация объекта для вычисления признаков
    vfe = OpenSMILE(opensmile_root_dir, opensmile_config_path)
    progresser = tqdm(
        iterable=range(0, len(data)),
        desc="calc audio features",
        total=len(data),
        unit="files",
    )

    feat, targets = [], []
    # Вычисление признаков голоса для каждого аудиофайла
    for i in progresser:
        clip = data[i]

        try:
            # Вычисление признаков голоса для аудиофайла
            voice_feat = vfe.process(clip.wav_rel_path)
            feat.append(voice_feat)
            targets.append(clip.labels) # классификация
            # targets.append([clip.valence, clip.arousal]) # регрессия
        except Exception as e:
            print(f"error calc voice features! {e}")
            data.remove(clip)

    print("feat count:", len(feat))
    print("Feature dimension per frame:", len(feat[0]))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim: int = 0):
    if pca_dim > 0:
        # Выполните сокращение размерности признаков с использованием PCA
        pca = PCA(n_components=pca_dim, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Перемешивание данных
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # Стандартизация признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Использование классификатора Random Forest
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest:")
    accuracy_fn.by_clips(y_pred_rf, classifier="Random Forest")

    # Использование классификатора SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_svm = grid_search.best_estimator_
    y_pred_svm = best_svm.predict(X_test)
    print("Support Vector Machine:")
    accuracy_fn.by_clips(y_pred_svm, classifier="Support Vector Machine")


def regression(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim: int = 0):
    if pca_dim > 0:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Использование регрессора Random Forest для valence и arousal отдельно
    rf_valence = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf_valence.fit(X_train, y_train[:, 0])
    y_pred_rf_valence = rf_valence.predict(X_test)

    rf_arousal = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf_arousal.fit(X_train, y_train[:, 1])
    y_pred_rf_arousal = rf_arousal.predict(X_test)

    y_pred_rf = np.vstack((y_pred_rf_valence, y_pred_rf_arousal)).T
    print("Random Forest Regressor:")
    accuracy_fn.by_clips(y_test, y_pred_rf)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    svr_valence = SVR()
    grid_search_valence = GridSearchCV(svr_valence, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_valence.fit(X_train, y_train[:, 0])
    best_svr_valence = grid_search_valence.best_estimator_
    y_pred_svr_valence = best_svr_valence.predict(X_test)

    svr_arousal = SVR()
    grid_search_arousal = GridSearchCV(svr_arousal, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_arousal.fit(X_train, y_train[:, 1])
    best_svr_arousal = grid_search_arousal.best_estimator_
    y_pred_svr_arousal = best_svr_arousal.predict(X_test)

    y_pred_svr = np.vstack((y_pred_svr_valence, y_pred_svr_arousal)).T
    print("Support Vector Regressor:")
    accuracy_fn.by_clips(y_test, y_pred_svr)

if __name__ == "__main__":

    experiment_name = "exp_09_01_2025"
    max_num_clips = 0
    use_dump = False  # dump для быстрой загрузки из файла
    is_regression = False  # регрессия или классификация

    base_dir = Path(r"C:\Users\zacep\Downloads\NeuralNetworksData\data.part1")
    if 1:
        # train_dataset_root = base_dir / "Ryerson/Video"
        # train_file_list = base_dir / "Ryerson/train_data_with_landmarks.txt"
        #
        # test_dataset_root = base_dir / "Ryerson/Video"
        # test_file_list = base_dir / "Ryerson/test_data_with_landmarks.txt"

        train_dataset_root = base_dir / "OMGEmotionChallenge/omg_TrainVideos/frames"
        train_file_list = base_dir / "OMGEmotionChallenge/omg_TrainVideos/train_data_with_landmarks.txt"

        test_dataset_root = base_dir / "OMGEmotionChallenge/omg_ValidVideos/frames"
        test_file_list = base_dir / "OMGEmotionChallenge/omg_ValidVideos/valid_data_with_landmarks.txt"
    else:
        train_dataset_root = base_dir / "OMGEmotionChallenge/omg_TrainVideos/frames"
        train_file_list = base_dir / "OMGEmotionChallenge/omg_TrainVideos/train_data_with_landmarks.txt"

        test_dataset_root = base_dir / "OMGEmotionChallenge/omg_ValidVideos/frames"
        test_file_list = base_dir / "OMGEmotionChallenge/omg_ValidVideos/valid_data_with_landmarks.txt"

    # Путь к библиотеке OpenSmile
    opensmile_root_dir = Path(r"C:\Users\zacep\Downloads\NeuralNetworksData\opensmile-2.3.0")
    opensmile_config_path = opensmile_root_dir / "config/avec2013.conf"
    # opensmile_config_path = opensmile_root_dir / "config/IS09_emotion.conf"

    if not use_dump:
        # Загрузка датасета
        train_data = get_data(train_dataset_root, train_file_list, max_num_clips=max_num_clips)
        test_data = get_data(test_dataset_root, test_file_list, max_num_clips=max_num_clips)

        # Вычисление признаков
        train_feat, train_targets = calc_features(train_data, opensmile_root_dir, opensmile_config_path)
        test_feat, test_targets = calc_features(test_data, opensmile_root_dir, opensmile_config_path)

        # Создание объекта для расчета метрик
        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name) # классификация
        # accuracy_fn = AccuracyRegression(test_data) # регрессия

        # Сохранение рассчитанных признаков на диск
        with open(experiment_name + '.pickle', 'wb') as f:
           pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        # Загрузка рассчитанных признаков с диска
        with open(experiment_name + ".pickle", "rb") as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(f)

    # run classifiers
    classification(
        train_feat,
        test_feat,
        train_targets,
        test_targets,
        accuracy_fn=accuracy_fn,
        pca_dim=100,
    )

    # run regression
    regression(
        train_feat,
        test_feat,
        train_targets,
        test_targets,
        accuracy_fn=accuracy_fn,
        pca_dim=100,
    )
