# ISVS/config.py
from dataclasses import dataclass

@dataclass
class CropConfig:
    run_name: str
    out_dir: str
    crop_mode: str
    encoder: str                 # "dino" or "mae" ← 追加
    model_name: str              # HFのモデル名
    vit_patch: int
    long_side: int
    pad_to_multiple: bool
    enable_dino_pca: bool        # MAEでもPCAは動くので名前はそのまま
    draw_query_box: bool
    multi_combine: str           # "mean"|"median"|"max"
    show_on_screen: bool
    flow_stride: int
    flow_min_score: float | None
    save_flow_figure: bool
    match_mode: str           # "oneway" | "mutual" | "mutual_topk"
    mutual_min_sim: float     # 相互一致の下限コサイン類似度 τ
    mutual_topk_K: int        # 逆方向で保持する上位K
    # 特徴抽出層の制御
    feat_mode: str        # "last" | "layer" | "avg_last_k"
    feat_layer: int       # feat_mode="layer" のとき使用（0起点, 負数で後ろから）
    feat_last_k: int      # feat_mode="avg_last_k" のとき使用（末尾からK層の平均）
    # ===== Ours 用（追加）=====
    ours_pca_dim: int | None = None   # n（NoneならPCA圧縮しない）
    ours_shallow_layer: int = 3       # DINOv3 の浅層（hidden_states の index。現状は3）
    #Ours の CLIP 側の層選択
    ours_clip_feat_mode: str = "layer"   # "last" | "layer" | "avg_last_k"
    ours_clip_feat_layer: int = 6       # CLIP 側の抽出層
    ours_clip_feat_last_k: int = 4       # avg_last_k 用
    # 追加：Oursの構成を選べるようにする
    ours_use_dino_deep: bool = True
    ours_use_dino_shallow: bool = True
    ours_use_clip: bool = True
    mad_k: float = 1.5
    # ===== Classic (SHIFT/ORB) 用 =====
    classic_max_features: int = 5000
    classic_ratio: float = 0.85
    classic_max_matches: int = 1000
    classic_use_cross_check: bool = False
    classic_use_mask: bool = True
    classic_sift_contrast: float = 0.01
    classic_sift_edge: float = 10.0
    classic_sift_sigma: float = 1.6
    classic_orb_fast_threshold: int = 5
    classic_orb_scale_factor: float = 1.2
    classic_orb_nlevels: int = 8
    classic_orb_wta_k: int = 2


_SCENARIOS = {
    # DINOv3で抽出
    "dino": CropConfig(
        run_name="dino_run",
        out_dir="outputs",
        crop_mode="scale_then_center_crop", # "direct_crop" | "scale_then_center_crop"
        encoder="dino",
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        vit_patch=16, # ViT-B/16のパッチサイズ
        long_side=1500, # 長辺リサイズ
        pad_to_multiple=True, # 16の倍数にパディング
        enable_dino_pca=True, # PCA可視化を有効にする
        draw_query_box=False, # クロップ領域を描画
        multi_combine="mean",      # 複数スケール特徴の統合方法
        show_on_screen=False,      # ダッシュボードを画面表示
        flow_stride=2,        # フロー可視化の間引き間隔
        flow_min_score=None,   # フロー可視化のスコア下限（Noneで全表示）
        save_flow_figure=True, # フロー可視化図を保存
        match_mode="oneway", # "oneway" | "mutual" | "mutual_topk"
        mutual_min_sim=0.0,    # 相互一致の下限コサイン類似度 τ
        mutual_topk_K=10,         # 逆方向で保持する上位K
        feat_mode="last",      # "last" | "layer" | "avg_last_k"
        feat_layer=1,   # 例: 中間層にしたいなら 6 など
        feat_last_k=4,   # 例: 末尾4層の平均
    ),
    # MAEで抽出
    "clip": CropConfig(
        run_name="clip_run",
        out_dir="outputs",
        crop_mode="scale_then_center_crop", # "direct_crop" | "scale_then_center_crop"
        encoder="clip",
        model_name="openai/clip-vit-large-patch14",  # 例: ViT-B/32 (patch=32)
        vit_patch=14,         # 参考値（実際はモデル設定から自動取得）
        long_side=1500,          # 解像度を上げたい場合は >0 に
        pad_to_multiple=True, # パッチ整合のため基本 True
        enable_dino_pca=True,
        draw_query_box=False,
        multi_combine="mean",
        show_on_screen=False,
        flow_stride=2,
        flow_min_score=None,
        save_flow_figure=True, # フロー可視化図を保存
        match_mode="oneway", # "oneway" | "mutual" | "mutual_topk"
        mutual_min_sim=0.0,
        mutual_topk_K=5,
        feat_mode="layer",      # "last" | "layer" | "avg_last_k"
        feat_layer=6,   # 例: 中間層にしたいなら 6 など
        feat_last_k=4,   # 例: 末尾4層の平均
    ),
    # ===== 追加：Ours =====
    "ours": CropConfig(
        run_name="ours_run",
        out_dir="outputs",
        crop_mode="scale_then_center_crop",
        encoder="ours",
        # "DINO|CLIP" 形式で渡す（encoders.py 側で分解）
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m|openai/clip-vit-large-patch14",
        vit_patch=16,          # Ours の基準グリッドは DINO 側（B/16）に合わせる
        long_side=1500,
        pad_to_multiple=True,
        enable_dino_pca=True,  # ダッシュボードのPCA可視化は残す（任意）
        draw_query_box=False,
        multi_combine="mean",
        show_on_screen=False,
        flow_stride=2,
        flow_min_score=None,      # フロー可視化のスコア下限（Noneで全表示）
        save_flow_figure=True,
        match_mode="oneway",  # Ours は相互TopK推奨
        mutual_min_sim=0.0,
        mutual_topk_K=30,

        # DINO深層の取り方
        feat_mode="last",
        feat_layer=9,
        feat_last_k=4,
        # DINO浅層の取り方
        ours_shallow_layer=6,      # DINOv3 の浅層（hidden_states の index。現状は3）
        # CLIP層の取り方
        ours_clip_feat_mode="layer",    # CLIP 側の特徴抽出設定
        ours_clip_feat_layer=6,   #CLIP 側の抽出層
        ours_clip_feat_last_k=4,  #CLIP 側の avg_last_k 用
        # PCA圧縮設定
        ours_pca_dim=None,         # PCA圧縮後の次元数（Noneで圧縮しない）
        # MADの閾値設定
        mad_k=1.5,                 # MADの閾値
        # Ours の構成選択
        ours_use_dino_deep=True,
        ours_use_dino_shallow=False,
        ours_use_clip=True,
    ),
    # ===== 追加：SIFT =====
    "sift": CropConfig(
        run_name="sift_run",
        out_dir="outputs",
        crop_mode="scale_then_center_crop",
        encoder="sift",
        model_name="opencv-sift",
        vit_patch=16,
        long_side=0,
        pad_to_multiple=False,
        enable_dino_pca=False,
        draw_query_box=False,
        multi_combine="mean",
        show_on_screen=False,
        flow_stride=1,
        flow_min_score=None,
        save_flow_figure=True,
        match_mode="oneway",
        mutual_min_sim=0.0,
        mutual_topk_K=10,
        feat_mode="last",
        feat_layer=-1,
        feat_last_k=1,
        classic_max_features=5000,
        classic_ratio=0.85,
        classic_max_matches=1000,
        classic_use_cross_check=False,
    ),
    # ===== 追加：ORB =====
    "orb": CropConfig(
        run_name="orb_run",
        out_dir="outputs",
        crop_mode="scale_then_center_crop",
        encoder="orb",
        model_name="opencv-orb",
        vit_patch=16,
        long_side=0,
        pad_to_multiple=False,
        enable_dino_pca=False,
        draw_query_box=False,
        multi_combine="mean",
        show_on_screen=False,
        flow_stride=1,
        flow_min_score=None,
        save_flow_figure=True,
        match_mode="oneway",
        mutual_min_sim=0.0,
        mutual_topk_K=10,
        feat_mode="last",
        feat_layer=-1,
        feat_last_k=1,
        classic_max_features=5000,
        classic_ratio=0.8,
        classic_max_matches=1000,
        classic_use_cross_check=False,
    ),
}

def get_config(scenario: str) -> CropConfig:
    if scenario == "shift":
        scenario = "sift"
    if scenario not in _SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario}'")
    return _SCENARIOS[scenario]

def list_scenarios():
    return list(_SCENARIOS.keys())
