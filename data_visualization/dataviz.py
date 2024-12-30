import sys
from sympy import false
sys.path.append("../")
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    print("frames count:", dataset_parser.get_dataset_size())
    return data

def calc_features(data, draw: bool = True):
    feat, targets = [], []
    for clip in data:
        if not clip.data_samples[0].labels in [7, 8]:
            continue

        for i, sample in enumerate(clip.data_samples):
            if i % 8 != 0:
                continue

            dist = []
            lm_ref = sample.landmarks[30]  # точка на носу
            # Расчет расстояний
            for j in range(len(sample.landmarks)):
                lm = sample.landmarks[j]
                dist.append(np.sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))

            # Угол между носом и горизонтальными границами губ
            vec1 = np.array(sample.landmarks[48]) - np.array(sample.landmarks[30])
            vec2 = np.array(sample.landmarks[54]) - np.array(sample.landmarks[30])
            mouth_nose_angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

            # Расстояние между вертикальными границами губ
            mouth_distance = np.sqrt((sample.landmarks[51][0] - sample.landmarks[57][0]) ** 2 +
                                   (sample.landmarks[51][1] - sample.landmarks[57][1]) ** 2)

            eye_mouth_distance = np.sqrt((sample.landmarks[36][0] - sample.landmarks[48][0]) ** 2 +
                                          (sample.landmarks[45][1] - sample.landmarks[54][1]) ** 2)

            feat.append(dist + [mouth_nose_angle, mouth_distance, eye_mouth_distance])
            targets.append(sample.labels)

            if draw:
                img = cv2.imread(sample.img_rel_path)
                for lm in sample.landmarks:
                    cv2.circle(img, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)
                cv2.imshow(sample.text_labels, img)
                cv2.waitKey(100)

    print("features count:", len(feat))
    print("targets count:", len(targets))
    x = np.asarray(feat, dtype=np.float32)
    print("features array shape:", x.shape)
    print(x[111])
    print(targets[111])
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)

def draw(points2D, targets, save=False):
    fig = plt.figure()
    plt.scatter(points2D[:, 0], points2D[:, 1], c=targets)
    plt.axis("off")
    if save:
        plt.savefig("scatter.png", bbox_inches="tight")
        plt.close(fig)
    else:
        fig.show()
        plt.pause(5)
        plt.close(fig)


def run_tsne(feat, targets, pca_dim=50, tsne_dim=2):
    if pca_dim > 0:
        feat = PCA(n_components=pca_dim).fit_transform(feat)

    distances2 = pairwise_distances(feat, metric="euclidean", squared=True)
    print("distances2", distances2.shape)
    # This return n x (n-1) prob array
    pij = manifold._t_sne._joint_probabilities(distances2, 30, False)
    print("pij", pij.shape)
    # Convert to n x n prob array
    pij = squareform(pij)
    print("pij", pij.shape)

    i, j = np.indices(pij.shape)
    print("i", i)
    i, j = i.ravel(), j.ravel()
    pij = pij.ravel().astype("float32")
    # Remove self-indices
    idx = i != j
    i, j, pij = i[idx], j[idx], pij[idx]

    model = torchTSNE(n_points=feat.shape[0], n_dim=tsne_dim)
    w = Wrapper(model, cuda=False, batchsize=feat.shape[0], epochs=30)

    for itr in range(5):
        w.fit(pij, i, j)
        # Visualize the results
        embed = model.logits.weight.cpu().data.numpy()
        draw(embed, targets)


if __name__ == "__main__":
    # dataset dir
    base_dir = Path(r"C:\Users\zacep\Downloads\NeuralNetworksData\data.part1")
    if 1:
        train_dataset_root = base_dir / "Ryerson/Video"
        train_file_list = base_dir / "Ryerson/train_data_with_landmarks.txt"
    elif 0:
        train_dataset_root = base_dir / "/AFEW-VA/crop"
        train_file_list = base_dir / "AFEW-VA/crop/train_data_with_landmarks.txt"
    elif 0:
        train_dataset_root = (
            base_dir / "OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames"
        )
        train_file_list = (
            base_dir
            / "/OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt"
        )

    # load dataset
    data = get_data(train_dataset_root, train_file_list, max_num_clips=0)

    # get features
    feat, targets = calc_features(data, draw=False)

    # run t-SNE
    run_tsne(feat, targets, pca_dim=0)
