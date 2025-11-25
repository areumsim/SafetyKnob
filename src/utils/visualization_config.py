# visualization_config.py

# ===============================
# 시각화 스타일 & 설정 모음
# ===============================

# matplotlib 스타일용 (rcParams에만 사용)
PLOT_STYLE = {
    "figure.figsize": (8, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 150,
    "figure.dpi": 100,
    "axes.grid": True,
    "font.family": "DejaVu Sans",  # 한글 폰트 대신 영어 폰트 사용
}

# 시각화 함수 내부에서만 쓰이는 기타 설정
PLOT_CONFIG = {
    "colors_by_category": {
        "Danger-Safe": "red",
        "Danger-Danger": "blue",
        "Safe-Safe": "green",
        "Random-Random": "gray",
    },
    "display": {
        "max_images": 3,
        "figsize_per_image": (3, 3),
    },
}

# ===============================
# PCA 화살표 스타일
# ===============================
PCA_ARROW_STYLE = {
    "width": 0.003,
    "head_width": 0.03,
    "length_includes_head": True,
    "color": "black",
    "alpha": 0.9,
}

# ===============================
# 기타 설정
# ===============================
DISPLAY_CONFIG = {
    "max_image_per_case": 3,
    "top_k_similar": 5,
    "sim_threshold_for_outlier": 0.9,
}
