"""
utils.py - 위험/안전 이미지 임베딩 분석을 위한 유틸리티 모듈

[주요 기능]
==================== 데이터 관련 ====================
- parse_key(filename): 파일명에서 유형/시나리오/케이스 정보 추출
  * 입력: 이미지 파일명 (예: '*_A26_Y-06_002.jpg')
  * 출력: (유형, 시나리오, 케이스) 튜플 (예: ('A26', '06', '002'))

- collect_by_key(folder): 폴더 내 이미지를 키별로 그룹화
  * 출력: {키: 이미지경로리스트} 형태의 딕셔너리
  * 사용: 위험/안전 이미지 쌍 분석의 기초

==================== 임베딩 분석 로직 ====================
- compute_avg_embeddings(pair_paths, embedder): 이미지 쌍의 평균 임베딩 계산
  * 출력: (danger_avg, safe_avg) 임베딩 벡터 쌍
  * 중요성: 개별 이미지가 아닌 그룹 단위 분석 가능

- get_avg_embeddings_by_key(key_dict, embedder): 키별 평균 임베딩 계산
  * 출력: {키: 평균벡터} 딕셔너리

- get_random_pairwise_diffs(avg_embs, num_pairs): 임의 쌍 간 차이 벡터 계산
  * 결과 해석: 랜덤 조합 벡터와 danger-safe 벡터 비교 가능

- get_typewise_mean_diffs(danger_dict, safe_dict, embedder): 유형별 평균 차이 벡터
  * 출력 파일: result_dir/typewise_mean_vectors.png
  * 해석: 각 유형별 특징적인 위험 방향성 시각화

- compare_typewise_to_global(type_means, global_mean_vec): 유형별 vs 전체 방향 비교
  * 콘솔 출력: 유형별 코사인 유사도 값
  * 의미: 높은 값 = 해당 유형이 전체 위험 방향과 일치


==================== 유사도 분석 및 이상치 탐색 ====================
- get_top_k_similar_pairs(keys_all, sims_all, danger_dict, safe_dict): 유사도 기준 상/하위 쌍 추출
  * 출력 파일: result_dir/low_similarity_pairs, result_dir/high_similarity_pairs
  * 해석: 가장 다른/비슷한 위험-안전 쌍 시각화

- find_weird_high_similarity_cases(): 높은 유사도 중 이상치 사례 추출
  * 출력 파일: result_dir/weird_high_similarity
  * 의의: 위험/안전 구분이 모호한 경계 케이스 발견

- project_danger_on_mean_axis(): 평균 위험 방향으로 투영한 극단적 위험 사례
  * 출력 파일: result_dir/extreme_danger_grid.png
  * 활용: 가장 '위험한' 케이스 시각화로 모델 이해


==================== 고급 분석 ====================
- extract_common_danger_directions(): PCA/t-SNE/UMAP 차원 축소 비교 분석
  * 출력 파일: result_dir/danger_direction_comparison.png
  * 장점: 비선형 구조 파악으로 복잡한 위험 패턴 식별

- analyze_model_specialization(): 모델별 위험 감지 효율성 비교
  * 출력 파일: result_dir/model_specialization_metrics.png
  * 평가 지표: 분리도, 코사인 유사도, 방향 정렬도

- analyze_feature_importance(): 중요 임베딩 차원 식별
  * 출력 파일: result_dir/feature_importance_pca.png, result_dir/top_features_umap.png
  * 의미: 위험/안전 구분에 중요한 특성을 파악하여 모델 이해

- visualize_attention_maps(): 모델 어텐션 패턴 시각화 (ViT 모델)
  * 출력 파일: result_dir/attention_maps.png, result_dir/attention_diff_*.png
  * 활용: 모델이 이미지의 어느 부분을 위험하다고 판단하는지 파악


==================== 결과 저장 ====================
    - save_results_np_and_json()


==================== 시각화 관련 ====================
: PCA 시각화 및 결과 저장
    - plot_cosine_similarity_distribution()
    - cluster_and_plot_embeddings()
    - plot_pca_diff()
    - plot_pca_diff_colored()
    - plot_with_mean_direction()
    - plot_typewise_mean_vectors_2d()
    - plot_diff_comparison()
    - plot_diff_comparison_with_mean_arrow()
    - show_image_pair_grid(): 이미지 쌍 시각화 (result_dir/pair_*.png)
    - save_extreme_danger_grid()

출력 파일 해석:
- pca_diff_by_type.png: 유형별 위험/안전 차이 분포 - 유사한 클러스터는 유사한 위험 패턴 의미
   - 밀집된 클러스터: 유사한 위험 특성을 가진 케이스
   - 이상치: 특이한 위험 패턴 또는 잠재적 분류 오류
- model_direction_consistency.png: 모델 간 일관성 - 높은 값은 모델 간 유사한 위험 인식 의미
   - 높은 일관성(>0.8): 신뢰할 수 있는 위험 패턴
   - 낮은 일관성(<0.5): 모델 간 불일치, 해당 케이스 재검토 필요
- feature_importance_pca.png: 중요 특성 분포 - 상위 특성이 위험/안전 구분에 중요
- attention_maps.png: 모델 주의 영역 - 위험 영역에 집중하는 패턴 확인
   - 붉은 영역: 모델이 주목하는 잠재적 위험 영역
   - 오버레이 비교: 위험/안전 이미지 간 주의 영역 차이


==================== 활용 방법 ====================
1. 기본 분석: 임베딩 추출 후 PCA/UMAP 시각화로 패턴 파악
2. 유형별 비교: 유형별 위험 방향 분석으로 위험 특성 이해
        - 위험 유형 분석: extract_common_danger_directions() → analyze_feature_importance()
        - 경계 케이스 발견: find_weird_high_similarity_cases() → visualize_attention_maps()
        - 극단적 위험 탐색: project_danger_on_mean_axis() → show_image_pair_grid()
4. 어텐션 분석: 위험 이미지에서 모델이 주목하는 영역 파악
"""

import os
import json
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import matplotlib
import random
import re
import torch
import cv2
from tqdm import tqdm

import umap
from transformers import AutoImageProcessor, AutoModel

from .visualization_config import (
    PLOT_STYLE,
    PLOT_CONFIG,
    PCA_ARROW_STYLE,
    DISPLAY_CONFIG,
)


# ==================== 파일 및 키 처리 ====================
def parse_key(filename):
    parts = filename.split("_")
    type_no = parts[1]  # A26
    scenario_no = parts[2].split("-")[1]  # Y-06 -> 06
    case_no = parts[3]  # 002
    return (type_no, scenario_no, case_no)


def collect_by_key(folder):
    data = defaultdict(list)
    for fname in os.listdir(folder):
        if fname.lower().endswith(".jpg"):
            key = parse_key(fname)
            path = os.path.join(folder, fname)
            data[key].append(path)
    return data


# ==================== 임베딩 평균 및 차이 계산 ====================
def compute_avg_embeddings(pair_paths, embedder):
    danger_paths, safe_paths = pair_paths
    danger_embs = embedder.extract_embeddings(danger_paths)
    safe_embs = embedder.extract_embeddings(safe_paths)
    return np.mean(danger_embs, axis=0), np.mean(safe_embs, axis=0)


def get_avg_embeddings_by_key(key_dict, embedder, sample_keys=None):
    avg_embeddings = {}
    keys = sample_keys or list(key_dict.keys())
    for key in keys:
        paths = key_dict[key]
        embs = embedder.extract_embeddings(paths)
        avg = np.mean(embs, axis=0)
        avg_embeddings[key] = avg
    return avg_embeddings


def get_random_pairwise_diffs(avg_embs, num_pairs=100):
    keys = list(avg_embs.keys())
    diffs = []
    for _ in range(num_pairs):
        a, b = random.sample(keys, 2)
        diff = avg_embs[a] - avg_embs[b]
        diffs.append(diff)
    return np.array(diffs)


def compare_typewise_to_global(type_means, global_mean_vec):
    print("📊 사고 유형별 방향성 (cosine similarity to global mean):")
    for t, vec in type_means.items():
        sim = cosine_similarity([vec], [global_mean_vec])[0][0]
        print(f" - {t}: {sim:.4f}")


def get_typewise_mean_diffs(danger_dict, safe_dict, embedder):
    type_vectors = {}
    for key in set(danger_dict.keys()) & set(safe_dict.keys()):
        t = key[0]
        danger_embs = embedder.extract_embeddings(danger_dict[key])
        safe_embs = embedder.extract_embeddings(safe_dict[key])
        diff = np.mean(danger_embs, axis=0) - np.mean(safe_embs, axis=0)
        if t not in type_vectors:
            type_vectors[t] = []
        type_vectors[t].append(diff)
    return {t: np.mean(vs, axis=0) for t, vs in type_vectors.items()}


# ==================== 유사도 분석 및 이상치 탐색 / Clustering & PCA ====================
# :  Danger-Safe Pair Analysis


def get_top_k_similar_pairs(
    keys_all, sims_all, danger_dict, safe_dict, k=5, reverse=False
):
    sims_array = np.array(sims_all)
    sorted_indices = np.argsort(sims_array)
    if reverse:
        sorted_indices = sorted_indices[::-1][:k]
    else:
        sorted_indices = sorted_indices[:k]

    selected = []
    for idx in sorted_indices:
        key = keys_all[idx]
        parts = key.split("_")
        key_tuple = (parts[0], parts[1], parts[2])
        selected.append(
            {
                "key": key,
                "similarity": float(sims_array[idx]),
                "danger_imgs": danger_dict.get(key_tuple, []),
                "safe_imgs": safe_dict.get(key_tuple, []),
            }
        )

    return selected


def find_weird_high_similarity_cases(
    keys_all,
    cosine_sims_all,
    vector_diffs,
    danger_dict,
    safe_dict,
    sim_threshold=0.9,
    top_k=5,
):
    labels, _ = get_cluster_labels_and_pca_coords(vector_diffs)
    outlier_scores = compute_outlier_scores(np.array(vector_diffs), labels)
    sims = np.array(cosine_sims_all)
    keys = np.array(keys_all)
    high_sim_mask = sims >= sim_threshold
    sims = sims[high_sim_mask]
    outlier_scores = outlier_scores[high_sim_mask]
    keys = keys[high_sim_mask]
    sorted_indices = np.argsort(outlier_scores)[-top_k:][::-1]
    selected = []
    for idx in sorted_indices:
        key = keys[idx]
        parts = key.split("_")
        key_tuple = (parts[0], parts[1], parts[2])
        selected.append(
            {
                "key": key,
                "similarity": float(sims[idx]),
                "outlier_score": float(outlier_scores[idx]),
                "danger_imgs": danger_dict.get(key_tuple, []),
                "safe_imgs": safe_dict.get(key_tuple, []),
            }
        )
    return selected


def get_cluster_labels_and_pca_coords(diff_vectors, n_clusters=3):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(diff_vectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(scaled)
    pca_coords = PCA(n_components=2).fit_transform(scaled)
    return labels, pca_coords


def compute_outlier_scores(diff_vectors, labels):
    kmeans = KMeans(n_clusters=len(set(labels)), random_state=42, n_init="auto")
    kmeans.fit(diff_vectors)
    centroids = kmeans.cluster_centers_
    distances = [
        np.linalg.norm(vec - centroids[labels[i]]) for i, vec in enumerate(diff_vectors)
    ]
    return np.array(distances)


###  위험성 투영 분석
def project_danger_on_mean_axis(danger_dict, embedder, mean_vector, top_k=5):
    projections = []
    for key in danger_dict.keys():
        embs = embedder.extract_embeddings(danger_dict[key])
        avg = np.mean(embs, axis=0)
        proj_value = np.dot(avg, mean_vector / np.linalg.norm(mean_vector))
        projections.append((key, proj_value))
    return sorted(projections, key=lambda x: -x[1])[:top_k]


# ==================== 결과 저장 ====================
def save_results_np_and_json(result_dir, vectors, sims, keys, prefix="filtered"):
    np.save(
        os.path.join(result_dir, f"danger_safe_diff_{prefix}.npy"), np.array(vectors)
    )
    np.save(os.path.join(result_dir, f"cosine_similarity_{prefix}.npy"), np.array(sims))
    with open(os.path.join(result_dir, f"{prefix}_keys.json"), "w") as f:
        json.dump(keys, f, indent=2)


# ==================== 시각화  ====================
def plot_cosine_similarity_distribution(
    similarities, result_dir, title="Cosine Similarity Distribution"
):
    plt.figure(figsize=(6, 4))
    sns.histplot(similarities, bins=50, kde=True, color="skyblue")
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(result_dir, "cosine_similarity_distribution.png")
    plt.savefig(path)
    print(f"📊 저장됨: {path}")
    plt.show()


def cluster_and_plot_embeddings(embeddings, title, result_dir, n_clusters=3):
    if len(embeddings) == 0:
        print(f"⚠️ No data for {title}")
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    reduced = PCA(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(6, 5))
    for label in np.unique(labels):
        idxs = labels == label
        plt.scatter(
            reduced[idxs, 0], reduced[idxs, 1], label=f"Cluster {label}", alpha=0.6
        )

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(
        result_dir, re.sub(r"[^a-zA-Z0-9_-]", "_", title).lower() + ".png"
    )
    plt.savefig(fig_path)
    print(f"📸 저장됨: {fig_path}")
    plt.show()


def plot_pca_diff(diff_vectors, labels=None, out_path=None):
    if len(diff_vectors) == 0:
        print("⚠️ vector_diffs is empty – skipping PCA.")
        return
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(diff_vectors)

    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], c="red", alpha=0.7)
    if labels:
        for i, label in enumerate(labels):
            plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=7)
    plt.title("PCA of Danger-Safe Embedding Differences")
    plt.grid(True)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
        print(f"📸 저장됨: {out_path}")
    plt.show()


def plot_pca_diff_colored(diff_vectors, filtered_keys, result_dir, color_by="type"):
    if len(diff_vectors) == 0:
        print("⚠️ vector_diffs is empty – skipping PCA.")
        return

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(diff_vectors)
    categories = []
    for key in filtered_keys:
        parts = key.split("_")
        if color_by == "type":
            categories.append(parts[0])
        elif color_by == "scenario":
            categories.append(parts[1])
        else:
            categories.append("Unknown")

    unique_categories = sorted(set(categories))
    cmap = matplotlib.colormaps.get_cmap("tab20")
    color_map = {cat: cmap(i) for i, cat in enumerate(unique_categories)}

    plt.figure(figsize=(7, 6))
    for i, vec in enumerate(reduced):
        plt.scatter(vec[0], vec[1], color=color_map[categories[i]], alpha=0.7, s=30)

    plt.title(f"PCA of Differences Colored by {color_by.capitalize()}")
    plt.grid(True)
    plt.tight_layout()

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=cat,
            markerfacecolor=color_map[cat],
            markersize=8,
        )
        for cat in unique_categories
    ]
    plt.legend(
        handles=handles,
        title=color_by.capitalize(),
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    filename = f"pca_diff_by_{color_by}.png"
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"📸 저장됨: {save_path}")
    plt.show()


def plot_with_mean_direction(diffs, out_path):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(diffs)
    mean_vec = np.mean(diffs, axis=0)
    mean_vec_2d = pca.transform(mean_vec.reshape(1, -1))[0]

    plt.figure(figsize=(7, 6))
    plt.scatter(
        reduced[:, 0], reduced[:, 1], alpha=0.6, s=30, color="red", label="Diff Vectors"
    )
    plt.arrow(
        0,
        0,
        mean_vec_2d[0],
        mean_vec_2d[1],
        color="black",
        width=0.003,
        head_width=0.03,
        length_includes_head=True,
        label="Mean Direction",
    )
    plt.title("Danger-Safe Differences with Mean Vector")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"📸 저장됨: {out_path}")
    plt.show()


def plot_typewise_mean_vectors_2d(typewise_means, out_path):
    type_labels = list(typewise_means.keys())
    vecs = np.stack([typewise_means[t] for t in type_labels])
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(vecs)

    plt.figure(figsize=(8, 7))
    for i, label in enumerate(type_labels):
        vec = vecs_2d[i]
        plt.arrow(
            0, 0, vec[0], vec[1], head_width=0.02, length_includes_head=True, alpha=0.8
        )
        plt.text(vec[0] * 1.05, vec[1] * 1.05, label, fontsize=9)

    plt.title("PCA: Typewise Mean Danger-Safe Vectors")
    plt.grid(True)
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"📸 저장됨: {out_path}")
    plt.show()


# def plot_diff_comparison(diffs_dict, result_dir):
#     colors = ["red", "blue", "green", "gray"]
#     plt.figure(figsize=(7, 6))

#     for i, (label, diffs) in enumerate(diffs_dict.items()):
#         reduced = PCA(n_components=2).fit_transform(diffs)
#         plt.scatter(
#             reduced[:, 0], reduced[:, 1], label=label, alpha=0.6, s=30, color=colors[i]
#         )

#     plt.title("PCA of Embedding Difference Vectors")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, "pca_diff_comparison.png"))
#     plt.show()


# def plot_diff_comparison_with_mean_arrow(
#     diffs_dict,
#     mean_vec_for_arrow,
#     result_dir,
#     mean_label="Mean Danger-Safe",
#     save_name="pca_diff_comparison_with_arrow.png",
# ):
#     colors = {
#         "Danger-Safe": "red",
#         "Danger-Danger": "blue",
#         "Safe-Safe": "green",
#         "Random-Random": "gray",
#     }

#     plt.figure(figsize=(8, 7))

#     # 전체 묶어서 PCA 학습 (동일 기준)
#     all_diffs = np.vstack(list(diffs_dict.values()))
#     pca = PCA(n_components=2)
#     reduced_all = pca.fit_transform(all_diffs)

#     # 개별 범주 마스킹 후 시각화
#     start_idx = 0
#     for label, diffs in diffs_dict.items():
#         n = len(diffs)
#         reduced = reduced_all[start_idx : start_idx + n]
#         plt.scatter(
#             reduced[:, 0],
#             reduced[:, 1],
#             alpha=0.6,
#             s=30,
#             label=label,
#             color=colors.get(label, "black"),
#         )
#         start_idx += n

#     # 화살표 (mean vector in PCA space)
#     mean_vec_2d = pca.transform(mean_vec_for_arrow.reshape(1, -1))[0]
#     plt.arrow(
#         0,
#         0,
#         mean_vec_2d[0],
#         mean_vec_2d[1],
#         color="black",
#         width=0.004,
#         head_width=0.04,
#         length_includes_head=True,
#         label=mean_label,
#     )

#     plt.title("PCA of Embedding Differences + Mean Direction")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     save_path = os.path.join(result_dir, save_name)
#     plt.savefig(save_path)
#     print(f"📸 저장됨: {save_path}")
#     plt.show()


def plot_diff_comparison(diffs_dict, result_dir):
    # plt.rcParams.update(PLOT_CONFIG)
    colors = PLOT_CONFIG["colors_by_category"]
    plt.figure(figsize=PLOT_STYLE["figure.figsize"])

    for label, diffs in diffs_dict.items():
        reduced = PCA(n_components=2).fit_transform(diffs)
        color = colors.get(label, "gray")
        plt.scatter(
            reduced[:, 0], reduced[:, 1], label=label, alpha=0.6, s=30, color=color
        )

    plt.title("PCA of Embedding Difference Vectors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "pca_diff_comparison.png"))
    plt.show()


def plot_diff_comparison_with_mean_arrow(
    diffs_dict,
    mean_vec_for_arrow,
    result_dir,
    save_name="pca_diff_comparison_with_arrow.png",
):
    colors = PLOT_CONFIG["colors_by_category"]
    plt.figure(figsize=PLOT_STYLE["figure.figsize"])

    # 전체 묶어서 PCA 학습 (동일 기준)
    all_diffs = np.vstack(list(diffs_dict.values()))
    pca = PCA(n_components=2)
    reduced_all = pca.fit_transform(all_diffs)

    # 개별 범주 마스킹 후 시각화
    start_idx = 0
    for label, diffs in diffs_dict.items():
        n = len(diffs)
        reduced = reduced_all[start_idx : start_idx + n]
        plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            alpha=0.6,
            s=30,
            label=label,
            color=colors.get(label, "gray"),
        )
        start_idx += n

    # 화살표 (mean vector in PCA space)
    mean_vec_2d = pca.transform(mean_vec_for_arrow.reshape(1, -1))[0]
    plt.arrow(
        0, 0, mean_vec_2d[0], mean_vec_2d[1], **PCA_ARROW_STYLE, label="Mean Direction"
    )

    plt.title("PCA of Embedding Differences + Mean Direction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(result_dir, save_name)
    plt.savefig(save_path)
    plt.show()


# ====================== Image Display ======================
def show_image_pair_grid(pairs, save_dir=None, max_imgs=None):
    max_imgs = max_imgs or DISPLAY_CONFIG.get("max_images", 2)
    figsize = DISPLAY_CONFIG.get("figsize_per_image", (3, 3))

    for entry in pairs:
        key = entry["key"]
        sim = entry["similarity"]
        danger_imgs = entry["danger_imgs"][:max_imgs]
        safe_imgs = entry["safe_imgs"][:max_imgs]
        n_rows = max(len(danger_imgs), len(safe_imgs))
        fig, axs = plt.subplots(
            n_rows, 2, figsize=(figsize[0] * 2, figsize[1] * n_rows)
        )
        fig.suptitle(f"{key} | Cosine Sim: {sim:.4f}", fontsize=14)

        for i in range(n_rows):
            for j, img_group in enumerate([danger_imgs, safe_imgs]):
                ax = axs[i, j] if n_rows > 1 else axs[j]
                if i < len(img_group):
                    img_path = img_group[i]
                    img = Image.open(img_path)
                    ax.imshow(img)

                    filename = os.path.basename(img_path)
                    is_danger = "_N-" in filename  # 또는 규칙에 따라 'DANGER' 포함 등
                    label = "Danger" if is_danger else "Safe"
                    ax.set_title(f"{label}\n{filename}", fontsize=8)
                ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"pair_{key}.png")
            plt.savefig(save_path)
            print(f"📸 저장됨: {save_path}")
        plt.show()
        plt.close()


def save_extreme_danger_grid(extreme_dangers, danger_dict, save_path, max_imgs=None):
    max_imgs = max_imgs or DISPLAY_CONFIG.get("max_images", 3)
    figsize = DISPLAY_CONFIG.get("figsize_per_image", (3, 3))

    n_cases = len(extreme_dangers)
    fig, axs = plt.subplots(
        n_cases, max_imgs, figsize=(figsize[0] * max_imgs, figsize[1] * n_cases)
    )

    for i, (key, score) in enumerate(extreme_dangers):
        img_paths = danger_dict[key][:max_imgs]
        for j in range(max_imgs):
            ax = axs[i, j] if n_cases > 1 else axs[j]
            if j < len(img_paths):
                img = Image.open(img_paths[j])
                ax.imshow(img)
                filename = os.path.basename(img_paths[j])
                is_danger = "_N-" in filename.lower()
                label = "Danger" if is_danger else "Safe"
                ax.set_title(f"{label}\n{filename}", fontsize=7)
            else:
                ax.axis("off")
            ax.axis("off")
        axs[i, 0].set_ylabel(f"{key}\nScore: {score:.3f}", fontsize=9)
    plt.suptitle("Extreme Danger Cases (Projection on Mean Direction)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📸 저장됨: {save_path}")
    plt.close()


# def save_extreme_danger_images(extreme_dangers, danger_dict, save_dir, max_imgs=3):
#     os.makedirs(save_dir, exist_ok=True)

#     for i, (key, score) in enumerate(extreme_dangers):
#         case_dir = os.path.join(save_dir, f"{i+1:02d}_{key[0]}_{key[1]}_{key[2]}_score_{score:.4f}")
#         os.makedirs(case_dir, exist_ok=True)

#         for j, img_path in enumerate(danger_dict[key][:max_imgs]):
#             try:
#                 img = Image.open(img_path).convert("RGB")
#                 img.save(os.path.join(case_dir, f"danger_{j+1}.jpg"))
#             except Exception as e:
#                 print(f"❌ 이미지 저장 실패: {img_path}, 에러: {e}")

#     print(f"📁 총 {len(extreme_dangers)}개의 extreme danger 이미지 저장 완료 → {save_dir}")


# ==================== 추가 분석1  ====================


# 1. 공통 위험 방향 추출: PCA, t-SNE, UMAP 비교 분석
def extract_common_danger_directions(vector_diffs, keys_all, result_dir):
    print("=== 공통 위험 방향 추출 (PCA/t-SNE/UMAP) 시작 ===")

    # 벡터 데이터 정규화
    scaler = StandardScaler()
    scaled_diffs = scaler.fit_transform(vector_diffs)

    # 1.1 PCA로 차원 축소
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_diffs)

    # 1.2 t-SNE로 차원 축소
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(scaled_diffs)

    # 1.3 UMAP으로 차원 축소
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_result = reducer.fit_transform(scaled_diffs)

    # 유형별 색상 지정
    # type_info = [k.split("_")[0] for k in keys_all]
    type_info = [k[0] if isinstance(k, tuple) else k.split("_")[0] for k in keys_all]
    unique_types = sorted(set(type_info))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_color_map = {t: colors[i] for i, t in enumerate(unique_types)}

    # 통합 시각화
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # PCA 시각화
    for t in unique_types:
        mask = np.array(type_info) == t
        axs[0].scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            c=[type_color_map[t]],
            label=t,
            alpha=0.7,
            s=40,
        )
    axs[0].set_title("PCA Projection")
    axs[0].grid(True)

    # t-SNE 시각화
    for t in unique_types:
        mask = np.array(type_info) == t
        axs[1].scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            c=[type_color_map[t]],
            label=t,
            alpha=0.7,
            s=40,
        )
    axs[1].set_title("t-SNE Projection")
    axs[1].grid(True)

    # UMAP 시각화
    for t in unique_types:
        mask = np.array(type_info) == t
        axs[2].scatter(
            umap_result[mask, 0],
            umap_result[mask, 1],
            c=[type_color_map[t]],
            label=t,
            alpha=0.7,
            s=40,
        )
    axs[2].set_title("UMAP Projection")
    axs[2].grid(True)

    # 범례는 마지막 plot에만 표시
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(unique_types) // 2,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "danger_direction_comparison.png"))
    plt.show()

    # 클러스터링 분석: 각 방법별 군집화 효율성 비교
    from sklearn.cluster import KMeans

    # 최적 클러스터 수 찾기 (2~10)
    k_range = range(2, 11)
    silhouette_scores = {"PCA": [], "t-SNE": [], "UMAP": []}

    for k in k_range:
        # PCA
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(pca_result)
        silhouette_scores["PCA"].append(silhouette_score(pca_result, labels))

        # t-SNE
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(tsne_result)
        silhouette_scores["t-SNE"].append(silhouette_score(tsne_result, labels))

        # UMAP
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(umap_result)
        silhouette_scores["UMAP"].append(silhouette_score(umap_result, labels))

    # 클러스터링 효율성 비교 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores["PCA"], "o-", label="PCA")
    plt.plot(k_range, silhouette_scores["t-SNE"], "s-", label="t-SNE")
    plt.plot(k_range, silhouette_scores["UMAP"], "^-", label="UMAP")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Clustering Efficiency Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "clustering_efficiency_comparison.png"))
    plt.show()

    # 최적 클러스터 수 선택
    best_method = max(silhouette_scores, key=lambda k: max(silhouette_scores[k]))
    best_k = k_range[np.argmax(silhouette_scores[best_method])]

    print(f"최적 군집화 방법: {best_method}, 최적 클러스터 수: {best_k}")

    # 최적 방법으로 클러스터링
    results = {"PCA": pca_result, "t-SNE": tsne_result, "UMAP": umap_result}

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    best_labels = kmeans.fit_predict(results[best_method])

    # 최종 클러스터링 시각화
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        results[best_method][:, 0],
        results[best_method][:, 1],
        c=best_labels,
        cmap="tab20",
        alpha=0.7,
        s=50,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Optimal Clustering: {best_method} with {best_k} clusters")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "optimal_danger_clusters.png"))
    plt.show()

    return {
        "pca": pca_result,
        "tsne": tsne_result,
        "umap": umap_result,
        "best_method": best_method,
        "best_k": best_k,
        "best_labels": best_labels,
    }


# # 2. 모델별 위험 감지 특화도 분석
# def analyze_model_specialization(danger_dict, safe_dict, model_types=None):
#     print("=== 모델별 위험 감지 특화도 분석 시작 ===")

#     if model_types is None:
#         model_types = ["dino", "clip", "siglip"]

#     # 모델 초기화
#     models = {}
#     for model_type in model_types:
#         config = {
#             "device": "cuda:0",
#             "model_type": model_type,
#             "cache_path": f"/workspace/prj_cctvPoc2_cv/tmp_ar_test/embedding_cache_{model_type}.pkl",
#         }
#         models[model_type] = get_embedder(config)

#     # 공통 키 찾기
#     common_keys = list(set(danger_dict.keys()) & set(safe_dict.keys()))

#     # 평가 메트릭 초기화
#     metrics = {
#         model_type: {
#             "separability": [],  # 유클리드 거리로 측정된 위험-안전 분리도
#             "cosine_similarity": [],  # 코사인 유사도
#             "alignment": [],  # 평균 위험 방향과의 정렬도
#         }
#         for model_type in model_types
#     }

#     # 평균 위험 방향 계산
#     mean_danger_vectors = {}

#     print("모델별 위험 감지 특화도 계산 중...")
#     for model_type, model in tqdm(models.items()):
#         vector_diffs = []

#         # 샘플링된 키로 테스트 (시간 단축)
#         sampled_keys = common_keys[:50] if len(common_keys) > 50 else common_keys

#         for key in sampled_keys:
#             danger_embs = model.extract_embeddings(danger_dict[key])
#             safe_embs = model.extract_embeddings(safe_dict[key])

#             danger_avg = np.mean(danger_embs, axis=0)
#             safe_avg = np.mean(safe_embs, axis=0)

#             # 벡터 차이
#             diff = danger_avg - safe_avg
#             vector_diffs.append(diff)

#             # 분리도 (유클리드 거리)
#             separability = np.linalg.norm(diff)
#             metrics[model_type]["separability"].append(separability)

#             # 코사인 유사도
#             from sklearn.metrics.pairwise import cosine_similarity

#             cos_sim = float(cosine_similarity([danger_avg], [safe_avg])[0][0])
#             metrics[model_type]["cosine_similarity"].append(cos_sim)

#         # 평균 위험 방향 계산
#         mean_danger_vector = np.mean(vector_diffs, axis=0)
#         mean_danger_vector = mean_danger_vector / np.linalg.norm(mean_danger_vector)
#         mean_danger_vectors[model_type] = mean_danger_vector

#         # 개별 벡터의 평균 방향과의 정렬도
#         for diff in vector_diffs:
#             normalized_diff = diff / np.linalg.norm(diff)
#             alignment = np.dot(normalized_diff, mean_danger_vector)
#             metrics[model_type]["alignment"].append(alignment)

#     # 모델간 평균 위험 방향 일관성
#     model_consistency = np.zeros((len(model_types), len(model_types)))
#     for i, model1 in enumerate(model_types):
#         for j, model2 in enumerate(model_types):
#             vec1 = mean_danger_vectors[model1]
#             vec2 = mean_danger_vectors[model2]
#             sim = np.dot(vec1, vec2)
#             model_consistency[i, j] = sim

#     # 일관성 히트맵 시각화
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(
#         model_consistency,
#         annot=True,
#         xticklabels=model_types,
#         yticklabels=model_types,
#         cmap="YlGnBu",
#         vmin=-1,
#         vmax=1,
#     )
#     plt.title("Model Consistency in Danger Direction")
#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, "model_direction_consistency.png"))
#     plt.show()

#     # 메트릭 요약
#     summary = {}
#     for model_type in model_types:
#         summary[model_type] = {
#             "avg_separability": np.mean(metrics[model_type]["separability"]),
#             "std_separability": np.std(metrics[model_type]["separability"]),
#             "avg_cosine_similarity": np.mean(metrics[model_type]["cosine_similarity"]),
#             "std_cosine_similarity": np.std(metrics[model_type]["cosine_similarity"]),
#             "avg_alignment": np.mean(metrics[model_type]["alignment"]),
#             "std_alignment": np.std(metrics[model_type]["alignment"]),
#         }

#     # 메트릭 시각화
#     metric_names = ["avg_separability", "avg_cosine_similarity", "avg_alignment"]
#     metric_labels = ["Separability", "Cosine Similarity", "Alignment"]

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     x = np.arange(len(model_types))
#     width = 0.25

#     for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
#         values = [summary[model][metric] for model in model_types]
#         errors = [summary[model][f"std_{metric[4:]}"] for model in model_types]

#         axs[i].bar(x, values, width, yerr=errors, capsize=5, alpha=0.7)
#         axs[i].set_title(label)
#         axs[i].set_xticks(x)
#         axs[i].set_xticklabels(model_types)
#         axs[i].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, "model_specialization_metrics.png"))
#     plt.show()

#     return {
#         "metrics": metrics,
#         "summary": summary,
#         "mean_vectors": mean_danger_vectors,
#         "consistency": model_consistency,
#     }


# 3. 특성 중요도 분석
def analyze_feature_importance(vector_diffs, keys_all):
    print("=== 특성 중요도 분석 시작 ===")

    # 1. 주성분 분석을 통한 중요 특성 추출
    pca = PCA()
    pca.fit(vector_diffs)

    # 주성분별 설명 분산
    explained_variance = pca.explained_variance_ratio_

    # 주성분별 특성 기여도
    components = pca.components_

    # 상위 n개 주성분 선택
    n_components = 5  # 상위 5개 주성분

    # 차원 별 중요도 계산 (전체 주성분에서의 기여도 합)
    feature_importance = np.zeros(vector_diffs.shape[1])
    for i in range(n_components):
        feature_importance += np.abs(components[i]) * explained_variance[i]

    # 중요도 정규화
    feature_importance = feature_importance / np.sum(feature_importance)

    # 상위 중요 특성 추출
    top_n = 20
    top_indices = np.argsort(-feature_importance)[:top_n]
    top_importance = feature_importance[top_indices]

    # 특성 중요도 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), top_importance, alpha=0.7)
    plt.xlabel("Feature Index (sorted by importance)")
    plt.ylabel("Normalized Importance")
    plt.title("Top 20 Important Features in Danger-Safe Differentiation")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(result_dir, "feature_importance_pca.png"))
    plt.show()

    # 2. 상위 주성분들의 설명력 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(explained_variance), "o-")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.title("Explained Variance by Principal Components")

    # 90% 설명력을 위한 주성분 수
    n_for_90 = np.where(np.cumsum(explained_variance) >= 0.9)[0][0] + 1
    plt.axhline(y=0.9, color="r", linestyle="--", alpha=0.5)
    plt.axvline(x=n_for_90, color="r", linestyle="--", alpha=0.5)
    plt.text(
        n_for_90 + 5, 0.9, f"90% variance needs {n_for_90} components", color="red"
    )

    plt.savefig(os.path.join(result_dir, "cumulative_variance_pca.png"))
    plt.show()

    # 3. 중요 특성 기반 클러스터링
    # 상위 특성만으로 데이터 재구성
    top_features = vector_diffs[:, top_indices]

    # UMAP으로 시각화
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(top_features)

    # 유형별 색상 지정
    type_info = [k.split("_")[0] for k in keys_all]
    unique_types = sorted(set(type_info))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_color_map = {t: colors[i] for i, t in enumerate(unique_types)}

    # 중요 특성 기반 UMAP 시각화
    plt.figure(figsize=(10, 8))
    for t in unique_types:
        mask = np.array(type_info) == t
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[type_color_map[t]],
            label=t,
            alpha=0.7,
            s=50,
        )

    plt.title("UMAP of Top Important Features")
    plt.legend(title="Type")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "top_features_umap.png"))
    plt.show()

    # 4. 주요 특성의 차원 분포 분석
    from collections import Counter

    # 특성의 원래 차원 추정 (임베딩 차원이 구조화되어 있다고 가정)
    dim_size = 1024  # 예상 임베딩 차원
    while dim_size * 3 < vector_diffs.shape[1]:  # RGB 채널 3개 가정
        dim_size *= 2

    dim_size = min(dim_size, vector_diffs.shape[1])

    # 각 특성이 어느 차원 그룹에 속하는지 계산
    dim_groups = {}
    for idx in top_indices:
        group_idx = idx // dim_size
        if group_idx not in dim_groups:
            dim_groups[group_idx] = []
        dim_groups[group_idx].append(idx)

    # 차원 그룹별 중요 특성 수 시각화
    dim_count = Counter({i: len(indices) for i, indices in dim_groups.items()})

    plt.figure(figsize=(8, 5))
    dim_labels = [f"Dim Group {i}" for i in dim_count.keys()]
    plt.bar(dim_labels, dim_count.values(), alpha=0.7)
    plt.xlabel("Dimension Group")
    plt.ylabel("Count of Important Features")
    plt.title("Distribution of Important Features across Dimension Groups")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "feature_dimension_distribution.png"))
    plt.show()

    return {
        "feature_importance": feature_importance,
        "top_indices": top_indices,
        "top_features": top_features,
        "explained_variance": explained_variance,
        "n_for_90": n_for_90,
        "embedding": embedding,
    }


### 위험 벡터 방향의 유형별 클러스터링 및 시각화
def analyze_danger_vector_clusters(vector_diffs, keys_all, n_clusters=5):
    """위험 벡터의 클러스터 분석 및 대표 이미지 추출"""
    # 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(vector_diffs)

    # 각 클러스터의 중심 벡터와 가장 가까운 이미지 찾기
    centers = kmeans.cluster_centers_
    representatives = []

    for i in range(n_clusters):
        cluster_mask = clusters == i
        if not any(cluster_mask):
            continue

        cluster_vectors = vector_diffs[cluster_mask]
        cluster_keys = np.array(keys_all)[cluster_mask]

        # 중심과 가장 가까운 벡터 찾기
        distances = np.linalg.norm(cluster_vectors - centers[i], axis=1)
        rep_idx = np.argmin(distances)
        representatives.append((i, cluster_keys[rep_idx]))

    return clusters, centers, representatives


###  임베딩 차원의 중요도 분석
def analyze_important_dimensions(vector_diffs, n_top=20):
    """위험-안전 차이를 가장 크게 설명하는 임베딩 차원 분석"""
    # 각 차원의 변동성 계산
    variances = np.var(vector_diffs, axis=0)

    # 가장 중요한 차원 탐색
    top_dims = np.argsort(-variances)[:n_top]
    importance = variances[top_dims] / np.sum(variances)

    return top_dims, importance


### 위험도 점수 예측 모델 개발
def build_danger_score_predictor(danger_dict, vector_diffs, safe_dict, embedder):
    """위험도 점수를 예측하는 모델 구축"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    # 평균 위험 벡터 계산 (build_danger_score_predictor에서 사용)
    mean_danger_vector = np.mean(vector_diffs, axis=0)

    # 데이터 준비
    X, y = [], []
    for key in danger_dict.keys():
        danger_embs = embedder.extract_embeddings(danger_dict[key])
        danger_avg = np.mean(danger_embs, axis=0)

        # 전체 평균 위험 벡터와의 내적으로 위험 점수 계산
        danger_score = np.dot(danger_avg, mean_danger_vector) / np.linalg.norm(
            mean_danger_vector
        )

        X.append(danger_avg)
        y.append(danger_score)

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 모델 학습
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    return model, (X_test, y_test)


### 시각화 방법
def plot_danger_embedding_landscape(vector_diffs, keys_all, types=None):
    """위험 임베딩 공간의 지형도 생성"""
    # UMAP으로 2D 축소
    import umap

    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine")
    embedding = reducer.fit_transform(vector_diffs)

    # 밀도 지형도 생성
    plt.figure(figsize=(10, 8))

    # 유형별 색상 지정
    type_info = [k.split("_")[0] for k in keys_all] if types is None else types
    unique_types = sorted(set(type_info))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_color_map = {t: colors[i] for i, t in enumerate(unique_types)}

    # 포인트 플롯
    for t in unique_types:
        mask = np.array(type_info) == t
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[type_color_map[t]],
            label=t,
            alpha=0.7,
            s=50,
        )

    # 밀도 등고선 추가
    from scipy.stats import gaussian_kde

    x, y = embedding[:, 0], embedding[:, 1]
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # 정규화된 밀도에 따라 포인트 크기 설정
    sizes = 50 * (density - density.min()) / density.ptp() + 20
    plt.scatter(x, y, c=density, cmap="viridis", alpha=0.3, s=sizes)

    plt.colorbar(label="Density")
    plt.title("UMAP Projection of Danger-Safe Differences")
    plt.legend(title="Type")

    return embedding, density


# 4. Attention Map 시각화
def visualize_attention_maps(danger_dict, safe_dict):
    print("=== Attention Map 시각화 시작 ===")

    # DINO 모델 로드 (ViT용 Attention Map 시각화에 최적)
    from transformers import AutoImageProcessor, AutoModel

    # 모델 초기화
    model_name = "facebook/dinov2-base"  # 더 작은 모델 사용
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 어텐션 맵 추출 함수
    def get_attention_map(image_path):
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # 어텐션 맵 추출
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

            # 모든 어텐션 헤드와 레이어에서 [CLS] 토큰의 어텐션 맵 추출
            attentions = outputs.attentions  # tuple of tensors (layers)

            # 마지막 레이어의 어텐션 맵 사용 (일반적으로 더 고수준 특성 포착)
            last_layer_attention = (
                attentions[-1].cpu().numpy()
            )  # [batch, num_heads, seq_len, seq_len]

            # 모든 헤드의 어텐션 평균
            attention_map = last_layer_attention[0].mean(axis=0)  # [seq_len, seq_len]

            # [CLS] 토큰이 다른 패치들에 주목하는 정도 ([CLS] 행 추출)
            cls_attention = attention_map[
                0, 1:
            ]  # 첫번째 행(CLS 토큰)의 다른 패치들에 대한 어텐션

        return cls_attention, image

    # 대표 이미지 쌍 선택
    selected_pairs = []
    common_keys = list(set(danger_dict.keys()) & set(safe_dict.keys()))

    # 각 유형별 대표 이미지
    type_keys = {}
    for key in common_keys:
        type_id = key[0]
        if type_id not in type_keys:
            type_keys[type_id] = key

    # 유형별로 최대 3개씩 선택
    for type_id, key in list(type_keys.items())[:3]:
        selected_pairs.append((key, danger_dict[key][0], safe_dict[key][0]))

    # 어텐션 맵 시각화
    fig, axs = plt.subplots(
        len(selected_pairs), 4, figsize=(20, 5 * len(selected_pairs))
    )

    for i, (key, danger_path, safe_path) in enumerate(selected_pairs):
        # Danger 이미지
        danger_attention, danger_img = get_attention_map(danger_path)

        # Safe 이미지
        safe_attention, safe_img = get_attention_map(safe_path)

        # 이미지 크기에 맞게 어텐션 맵 리사이징
        patch_size = 16  # ViT의 패치 크기
        num_patches = int(np.sqrt(len(danger_attention)))
        attention_size = num_patches * patch_size

        # 어텐션 맵 재구성 (1차원 -> 2차원)
        danger_attention_map = danger_attention.reshape(num_patches, num_patches)
        safe_attention_map = safe_attention.reshape(num_patches, num_patches)

        # 시각화용 리사이징
        danger_img_np = np.array(danger_img.resize((attention_size, attention_size)))
        safe_img_np = np.array(safe_img.resize((attention_size, attention_size)))

        # 어텐션 맵 업샘플링
        danger_attention_resized = cv2.resize(
            danger_attention_map, (attention_size, attention_size)
        )
        safe_attention_resized = cv2.resize(
            safe_attention_map, (attention_size, attention_size)
        )

        # 히트맵+이미지 오버레이
        danger_heatmap = cv2.applyColorMap(
            np.uint8(255 * danger_attention_resized), cv2.COLORMAP_JET
        )
        danger_heatmap = cv2.cvtColor(danger_heatmap, cv2.COLOR_BGR2RGB)
        danger_overlay = danger_heatmap * 0.5 + danger_img_np * 0.5
        danger_overlay = danger_overlay.astype(np.uint8)

        safe_heatmap = cv2.applyColorMap(
            np.uint8(255 * safe_attention_resized), cv2.COLORMAP_JET
        )
        safe_heatmap = cv2.cvtColor(safe_heatmap, cv2.COLOR_BGR2RGB)
        safe_overlay = safe_heatmap * 0.5 + safe_img_np * 0.5
        safe_overlay = safe_overlay.astype(np.uint8)

        # 두 어텐션 맵의 차이
        diff_attention = danger_attention_map - safe_attention_map
        diff_attention_resized = cv2.resize(
            diff_attention, (attention_size, attention_size)
        )

        # 차이 맵 시각화 (Red = Danger 중요, Blue = Safe 중요)
        diff_heatmap = np.zeros((attention_size, attention_size, 3), dtype=np.uint8)

        # 양수 값(Danger에서 더 중요)은 빨간색으로
        positive_mask = diff_attention_resized > 0
        diff_heatmap[positive_mask] = [
            0,
            0,
            np.uint8(
                255
                * diff_attention_resized[positive_mask]
                / diff_attention_resized.max()
            ),
        ]

        # 음수 값(Safe에서 더 중요)은 파란색으로
        negative_mask = diff_attention_resized < 0
        diff_heatmap[negative_mask] = [
            np.uint8(
                255
                * -diff_attention_resized[negative_mask]
                / diff_attention_resized.min()
            ),
            0,
            0,
        ]

        # 각 이미지 시각화
        if len(selected_pairs) > 1:
            axs[i, 0].imshow(danger_img)
            axs[i, 0].set_title(f"Danger Image ({key})")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(danger_overlay)
            axs[i, 1].set_title("Danger Attention")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(safe_img)
            axs[i, 2].set_title("Safe Image")
            axs[i, 2].axis("off")

            axs[i, 3].imshow(safe_overlay)
            axs[i, 3].set_title("Safe Attention")
            axs[i, 3].axis("off")
        else:
            axs[0].imshow(danger_img)
            axs[0].set_title(f"Danger Image ({key})")
            axs[0].axis("off")

            axs[1].imshow(danger_overlay)
            axs[1].set_title("Danger Attention")
            axs[1].axis("off")

            axs[2].imshow(safe_img)
            axs[2].set_title("Safe Image")
            axs[2].axis("off")

            axs[3].imshow(safe_overlay)
            axs[3].set_title("Safe Attention")
            axs[3].axis("off")
        # 차이 맵 별도 시각화
        plt.figure(figsize=(8, 8))
        plt.imshow(diff_heatmap)
        plt.title(f"Attention Difference (Danger-Safe) for {key}")
        plt.colorbar(label="Attention Difference")
        plt.savefig(
            os.path.join(result_dir, f"attention_diff_{key[0]}_{key[1]}_{key[2]}.png")
        )
        plt.close()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "attention_maps.png"))
    plt.show()

    # PCA를 통한 어텐션 패턴 분석
    print("어텐션 패턴 분석 중...")

    # 전체 샘플에서 임의로 30개 선택하여 어텐션 맵 추출
    sample_size = min(30, len(common_keys))
    sampled_keys = random.sample(common_keys, sample_size)

    danger_attentions = []
    safe_attentions = []

    for key in tqdm(sampled_keys):
        danger_img = danger_dict[key][0]
        safe_img = safe_dict[key][0]

        danger_att, _ = get_attention_map(danger_img)
        safe_att, _ = get_attention_map(safe_img)

        danger_attentions.append(danger_att)
        safe_attentions.append(safe_att)

    # 어텐션 패턴 PCA
    danger_attentions = np.array(danger_attentions)
    safe_attentions = np.array(safe_attentions)

    # PCA로 주요 어텐션 패턴 추출
    pca = PCA(n_components=2)
    danger_att_pca = pca.fit_transform(danger_attentions)
    safe_att_pca = pca.transform(safe_attentions)

    # 어텐션 PCA 시각화
    plt.figure(figsize=(10, 8))
    plt.scatter(
        danger_att_pca[:, 0],
        danger_att_pca[:, 1],
        c="red",
        label="Danger",
        alpha=0.7,
        s=100,
    )
    plt.scatter(
        safe_att_pca[:, 0], safe_att_pca[:, 1], c="blue", label="Safe", alpha=0.7, s=100
    )

    # 라인으로 연결
    for i in range(len(danger_att_pca)):
        plt.plot(
            [danger_att_pca[i, 0], safe_att_pca[i, 0]],
            [danger_att_pca[i, 1], safe_att_pca[i, 1]],
            "k-",
            alpha=0.3,
        )

    plt.title("PCA of Attention Patterns")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "attention_patterns_pca.png"))
    plt.show()

    return {
        "selected_pairs": selected_pairs,
        "danger_attentions": danger_attentions,
        "safe_attentions": safe_attentions,
        "danger_att_pca": danger_att_pca,
        "safe_att_pca": safe_att_pca,
    }


# 3. 특성 중요도 분석
def analyze_feature_importance(vector_diffs, keys_all, result_dir):
    print("=== 특성 중요도 분석 시작 ===")

    # 1. 주성분 분석을 통한 중요 특성 추출
    pca = PCA()
    pca.fit(vector_diffs)

    # 주성분별 설명 분산
    explained_variance = pca.explained_variance_ratio_

    # 주성분별 특성 기여도
    components = pca.components_

    # 상위 n개 주성분 선택
    n_components = 5  # 상위 5개 주성분

    # 차원 별 중요도 계산 (전체 주성분에서의 기여도 합)
    feature_importance = np.zeros(vector_diffs.shape[1])
    for i in range(n_components):
        feature_importance += np.abs(components[i]) * explained_variance[i]

    # 중요도 정규화
    feature_importance = feature_importance / np.sum(feature_importance)

    # 상위 중요 특성 추출
    top_n = 20
    top_indices = np.argsort(-feature_importance)[:top_n]
    top_importance = feature_importance[top_indices]

    # 특성 중요도 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), top_importance, alpha=0.7)
    plt.xlabel("Feature Index (sorted by importance)")
    plt.ylabel("Normalized Importance")
    plt.title("Top 20 Important Features in Danger-Safe Differentiation")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(result_dir, "feature_importance_pca.png"))
    plt.show()

    # 2. 상위 주성분들의 설명력 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(explained_variance), "o-")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.title("Explained Variance by Principal Components")

    # 90% 설명력을 위한 주성분 수
    n_for_90 = np.where(np.cumsum(explained_variance) >= 0.9)[0][0] + 1
    plt.axhline(y=0.9, color="r", linestyle="--", alpha=0.5)
    plt.axvline(x=n_for_90, color="r", linestyle="--", alpha=0.5)
    plt.text(
        n_for_90 + 5, 0.9, f"90% variance needs {n_for_90} components", color="red"
    )

    plt.savefig(os.path.join(result_dir, "cumulative_variance_pca.png"))
    plt.show()

    # 3. 중요 특성 기반 클러스터링
    # 상위 특성만으로 데이터 재구성
    top_features = vector_diffs[:, top_indices]

    # UMAP으로 시각화
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(top_features)

    # 유형별 색상 지정
    type_info = [k.split("_")[0] for k in keys_all]
    unique_types = sorted(set(type_info))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_color_map = {t: colors[i] for i, t in enumerate(unique_types)}

    # 중요 특성 기반 UMAP 시각화
    plt.figure(figsize=(10, 8))
    for t in unique_types:
        mask = np.array(type_info) == t
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[type_color_map[t]],
            label=t,
            alpha=0.7,
            s=50,
        )

    plt.title("UMAP of Top Important Features")
    plt.legend(title="Type")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "top_features_umap.png"))
    plt.show()

    # 4. 주요 특성의 차원 분포 분석

    # 특성의 원래 차원 추정 (임베딩 차원이 구조화되어 있다고 가정)
    dim_size = 1024  # 예상 임베딩 차원
    while dim_size * 3 < vector_diffs.shape[1]:  # RGB 채널 3개 가정
        dim_size *= 2

    dim_size = min(dim_size, vector_diffs.shape[1])

    # 각 특성이 어느 차원 그룹에 속하는지 계산
    dim_groups = {}
    for idx in top_indices:
        group_idx = idx // dim_size
        if group_idx not in dim_groups:
            dim_groups[group_idx] = []
        dim_groups[group_idx].append(idx)

    # 차원 그룹별 중요 특성 수 시각화
    dim_count = Counter({i: len(indices) for i, indices in dim_groups.items()})

    plt.figure(figsize=(8, 5))
    dim_labels = [f"Dim Group {i}" for i in dim_count.keys()]
    plt.bar(dim_labels, dim_count.values(), alpha=0.7)
    plt.xlabel("Dimension Group")
    plt.ylabel("Count of Important Features")
    plt.title("Distribution of Important Features across Dimension Groups")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "feature_dimension_distribution.png"))
    plt.show()

    return {
        "feature_importance": feature_importance,
        "top_indices": top_indices,
        "top_features": top_features,
        "explained_variance": explained_variance,
        "n_for_90": n_for_90,
        "embedding": embedding,
    }


### 위험 벡터 방향의 유형별 클러스터링 및 시각화
def analyze_danger_vector_clusters(vector_diffs, keys_all, n_clusters=5):
    """위험 벡터의 클러스터 분석 및 대표 이미지 추출"""
    # 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(vector_diffs)

    # 각 클러스터의 중심 벡터와 가장 가까운 이미지 찾기
    centers = kmeans.cluster_centers_
    representatives = []

    for i in range(n_clusters):
        cluster_mask = clusters == i
        if not any(cluster_mask):
            continue

        cluster_vectors = vector_diffs[cluster_mask]
        cluster_keys = np.array(keys_all)[cluster_mask]

        # 중심과 가장 가까운 벡터 찾기
        distances = np.linalg.norm(cluster_vectors - centers[i], axis=1)
        rep_idx = np.argmin(distances)
        representatives.append((i, cluster_keys[rep_idx]))

    return clusters, centers, representatives


###  임베딩 차원의 중요도 분석
def analyze_important_dimensions(vector_diffs, n_top=20):
    """위험-안전 차이를 가장 크게 설명하는 임베딩 차원 분석"""
    # 각 차원의 변동성 계산
    variances = np.var(vector_diffs, axis=0)

    # 가장 중요한 차원 탐색
    top_dims = np.argsort(-variances)[:n_top]
    importance = variances[top_dims] / np.sum(variances)

    return top_dims, importance


### 위험도 점수 예측 모델 개발
def build_danger_score_predictor(danger_dict, vector_diffs, safe_dict, embedder):
    """위험도 점수를 예측하는 모델 구축"""

    # 평균 위험 벡터 계산 (build_danger_score_predictor에서 사용)
    mean_danger_vector = np.mean(vector_diffs, axis=0)

    # 데이터 준비
    X, y = [], []
    for key in danger_dict.keys():
        danger_embs = embedder.extract_embeddings(danger_dict[key])
        danger_avg = np.mean(danger_embs, axis=0)

        # 전체 평균 위험 벡터와의 내적으로 위험 점수 계산
        danger_score = np.dot(danger_avg, mean_danger_vector) / np.linalg.norm(
            mean_danger_vector
        )

        X.append(danger_avg)
        y.append(danger_score)

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 모델 학습
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    return model, (X_test, y_test)


### 시각화 방법
def plot_danger_embedding_landscape(vector_diffs, keys_all, types=None):
    """위험 임베딩 공간의 지형도 생성"""
    # UMAP으로 2D 축소
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine")
    embedding = reducer.fit_transform(vector_diffs)

    # 밀도 지형도 생성
    plt.figure(figsize=(10, 8))

    # 유형별 색상 지정
    type_info = [k.split("_")[0] for k in keys_all] if types is None else types
    unique_types = sorted(set(type_info))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_color_map = {t: colors[i] for i, t in enumerate(unique_types)}

    # 포인트 플롯
    for t in unique_types:
        mask = np.array(type_info) == t
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[type_color_map[t]],
            label=t,
            alpha=0.7,
            s=50,
        )

    # 밀도 등고선 추가
    x, y = embedding[:, 0], embedding[:, 1]
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # 정규화된 밀도에 따라 포인트 크기 설정
    sizes = 50 * (density - density.min()) / density.ptp() + 20
    plt.scatter(x, y, c=density, cmap="viridis", alpha=0.3, s=sizes)

    plt.colorbar(label="Density")
    plt.title("UMAP Projection of Danger-Safe Differences")
    plt.legend(title="Type")

    return embedding, density


# 4. Attention Map 시각화
def visualize_attention_maps(danger_dict, safe_dict, result_dir):
    print("=== Attention Map 시각화 시작 ===")
    # DINO 모델 로드 (ViT용 Attention Map 시각화에 최적)
    # 모델 초기화
    model_name = "facebook/dinov2-base"  # 더 작은 모델 사용
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 어텐션 맵 추출 함수
    def get_attention_map(image_path):
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # 어텐션 맵 추출
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

            # 모든 어텐션 헤드와 레이어에서 [CLS] 토큰의 어텐션 맵 추출
            attentions = outputs.attentions  # tuple of tensors (layers)

            # 마지막 레이어의 어텐션 맵 사용 (일반적으로 더 고수준 특성 포착)
            last_layer_attention = (
                attentions[-1].cpu().numpy()
            )  # [batch, num_heads, seq_len, seq_len]

            # 모든 헤드의 어텐션 평균
            attention_map = last_layer_attention[0].mean(axis=0)  # [seq_len, seq_len]

            # [CLS] 토큰이 다른 패치들에 주목하는 정도 ([CLS] 행 추출)
            cls_attention = attention_map[
                0, 1:
            ]  # 첫번째 행(CLS 토큰)의 다른 패치들에 대한 어텐션

        return cls_attention, image

    # 대표 이미지 쌍 선택
    selected_pairs = []
    common_keys = list(set(danger_dict.keys()) & set(safe_dict.keys()))

    # 각 유형별 대표 이미지
    type_keys = {}
    for key in common_keys:
        type_id = key[0]
        if type_id not in type_keys:
            type_keys[type_id] = key

    # 유형별로 최대 3개씩 선택
    for type_id, key in list(type_keys.items())[:3]:
        selected_pairs.append((key, danger_dict[key][0], safe_dict[key][0]))

    # 어텐션 맵 시각화
    fig, axs = plt.subplots(
        len(selected_pairs), 4, figsize=(20, 5 * len(selected_pairs))
    )

    for i, (key, danger_path, safe_path) in enumerate(selected_pairs):
        # Danger 이미지
        danger_attention, danger_img = get_attention_map(danger_path)

        # Safe 이미지
        safe_attention, safe_img = get_attention_map(safe_path)

        # 이미지 크기에 맞게 어텐션 맵 리사이징
        patch_size = 16  # ViT의 패치 크기
        num_patches = int(np.sqrt(len(danger_attention)))
        attention_size = num_patches * patch_size

        # 어텐션 맵 재구성 (1차원 -> 2차원)
        danger_attention_map = danger_attention.reshape(num_patches, num_patches)
        safe_attention_map = safe_attention.reshape(num_patches, num_patches)

        # 시각화용 리사이징
        danger_img_np = np.array(danger_img.resize((attention_size, attention_size)))
        safe_img_np = np.array(safe_img.resize((attention_size, attention_size)))

        # 어텐션 맵 업샘플링
        danger_attention_resized = cv2.resize(
            danger_attention_map, (attention_size, attention_size)
        )
        safe_attention_resized = cv2.resize(
            safe_attention_map, (attention_size, attention_size)
        )

        # 히트맵+이미지 오버레이
        danger_heatmap = cv2.applyColorMap(
            np.uint8(255 * danger_attention_resized), cv2.COLORMAP_JET
        )
        danger_heatmap = cv2.cvtColor(danger_heatmap, cv2.COLOR_BGR2RGB)
        danger_overlay = danger_heatmap * 0.5 + danger_img_np * 0.5
        danger_overlay = danger_overlay.astype(np.uint8)

        safe_heatmap = cv2.applyColorMap(
            np.uint8(255 * safe_attention_resized), cv2.COLORMAP_JET
        )
        safe_heatmap = cv2.cvtColor(safe_heatmap, cv2.COLOR_BGR2RGB)
        safe_overlay = safe_heatmap * 0.5 + safe_img_np * 0.5
        safe_overlay = safe_overlay.astype(np.uint8)

        # 두 어텐션 맵의 차이
        diff_attention = danger_attention_map - safe_attention_map
        diff_attention_resized = cv2.resize(
            diff_attention, (attention_size, attention_size)
        )

        # 차이 맵 시각화 (Red = Danger 중요, Blue = Safe 중요)
        diff_heatmap = np.zeros((attention_size, attention_size, 3), dtype=np.uint8)

        # 양수 값(Danger에서 더 중요)은 빨간색으로
        positive_mask = diff_attention_resized > 0
        if diff_attention_resized.max() != 0:
            pos_values = np.uint8(
                255
                * diff_attention_resized[positive_mask]
                / diff_attention_resized.max()
            )
        else:
            pos_values = np.zeros(
                diff_attention_resized[positive_mask].shape, dtype=np.uint8
            )
        diff_heatmap[positive_mask] = np.stack(
            [
                pos_values,
                np.zeros_like(pos_values, dtype=np.uint8),
                np.zeros_like(pos_values, dtype=np.uint8),
            ],
            axis=-1,
        )

        # 음수 값 (안전 이미지에서 더 중요한 부분): 파란색 채널에 값 할당
        negative_mask = diff_attention_resized < 0
        if diff_attention_resized.min() != 0:
            neg_values = np.uint8(
                255
                * np.abs(diff_attention_resized[negative_mask])
                / np.abs(diff_attention_resized.min())
            )
        else:
            neg_values = np.zeros(
                diff_attention_resized[negative_mask].shape, dtype=np.uint8
            )
        diff_heatmap[negative_mask] = np.stack(
            [
                np.zeros_like(neg_values, dtype=np.uint8),
                np.zeros_like(neg_values, dtype=np.uint8),
                neg_values,
            ],
            axis=-1,
        )

        # 각 이미지 시각화
        if len(selected_pairs) > 1:
            axs[i, 0].imshow(danger_img)
            axs[i, 0].set_title(f"Danger Image ({key})")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(danger_overlay)
            axs[i, 1].set_title("Danger Attention")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(safe_img)
            axs[i, 2].set_title("Safe Image")
            axs[i, 2].axis("off")

            axs[i, 3].imshow(safe_overlay)
            axs[i, 3].set_title("Safe Attention")
            axs[i, 3].axis("off")
        else:
            axs[0].imshow(danger_img)
            axs[0].set_title(f"Danger Image ({key})")
            axs[0].axis("off")

            axs[1].imshow(danger_overlay)
            axs[1].set_title("Danger Attention")
            axs[1].axis("off")

            axs[2].imshow(safe_img)
            axs[2].set_title("Safe Image")
            axs[2].axis("off")

            axs[3].imshow(safe_overlay)
            axs[3].set_title("Safe Attention")
            axs[3].axis("off")
        # 차이 맵 별도 시각화
        plt.figure(figsize=(8, 8))
        plt.imshow(diff_heatmap)
        plt.title(f"Attention Difference (Danger-Safe) for {key}")
        plt.colorbar(label="Attention Difference")
        plt.savefig(
            os.path.join(result_dir, f"attention_diff_{key[0]}_{key[1]}_{key[2]}.png")
        )
        plt.close()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "attention_maps.png"))
    plt.show()

    # PCA를 통한 어텐션 패턴 분석
    print("어텐션 패턴 분석 중...")

    # 전체 샘플에서 임의로 30개 선택하여 어텐션 맵 추출
    sample_size = min(30, len(common_keys))
    sampled_keys = random.sample(common_keys, sample_size)

    danger_attentions = []
    safe_attentions = []

    for key in tqdm(sampled_keys):
        danger_img_path = danger_dict[key][0]
        safe_img_path = safe_dict[key][0]

        danger_att, _ = get_attention_map(danger_img_path)
        safe_att, _ = get_attention_map(safe_img_path)

        danger_attentions.append(danger_att)
        safe_attentions.append(safe_att)

    # 어텐션 패턴 PCA
    danger_attentions = np.array(danger_attentions)
    safe_attentions = np.array(safe_attentions)

    # PCA로 주요 어텐션 패턴 추출
    pca = PCA(n_components=2)
    danger_att_pca = pca.fit_transform(danger_attentions)
    safe_att_pca = pca.transform(safe_attentions)

    # 어텐션 PCA 시각화
    plt.figure(figsize=(10, 8))
    plt.scatter(
        danger_att_pca[:, 0],
        danger_att_pca[:, 1],
        c="red",
        label="Danger",
        alpha=0.7,
        s=100,
    )
    plt.scatter(
        safe_att_pca[:, 0], safe_att_pca[:, 1], c="blue", label="Safe", alpha=0.7, s=100
    )

    # 라인으로 연결
    for i in range(len(danger_att_pca)):
        plt.plot(
            [danger_att_pca[i, 0], safe_att_pca[i, 0]],
            [danger_att_pca[i, 1], safe_att_pca[i, 1]],
            "k-",
            alpha=0.3,
        )

    plt.title("PCA of Attention Patterns")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "attention_patterns_pca.png"))
    plt.show()

    return {
        "selected_pairs": selected_pairs,
        "danger_attentions": danger_attentions,
        "safe_attentions": safe_attentions,
        "danger_att_pca": danger_att_pca,
        "safe_att_pca": safe_att_pca,
    }


# ==================== 추가 분석2  ====================
# PCA 후 danger-safe 점들을 선으로 연결
def plot_embedding_trajectories(danger_dict, safe_dict, embedder, result_dir):
    danger_vecs, safe_vecs = [], []
    keys = list(set(danger_dict.keys()) & set(safe_dict.keys()))

    for key in keys:
        d_emb = np.mean(embedder.extract_embeddings(danger_dict[key]), axis=0)
        s_emb = np.mean(embedder.extract_embeddings(safe_dict[key]), axis=0)
        danger_vecs.append(d_emb)
        safe_vecs.append(s_emb)

    danger_vecs = np.array(danger_vecs)
    safe_vecs = np.array(safe_vecs)
    pca = PCA(n_components=2)
    all_vecs = np.vstack([danger_vecs, safe_vecs])
    reduced = pca.fit_transform(all_vecs)
    d2 = reduced[: len(keys)]
    s2 = reduced[len(keys) :]

    plt.figure(figsize=(10, 8))
    for i in range(len(keys)):
        plt.arrow(
            d2[i, 0],
            d2[i, 1],
            s2[i, 0] - d2[i, 0],
            s2[i, 1] - d2[i, 1],
            head_width=0.05,
            alpha=0.5,
        )
    plt.title("Danger → Safe Embedding Trajectories (PCA)")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "embedding_trajectories.png"))
    plt.show()


# ==================== 추가 분석3  ====================
