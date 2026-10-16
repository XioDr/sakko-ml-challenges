"""
Shared feature pipeline for Sakko (ICDS 9th hackathon) challenges 1/2/3.

Builds one feature table per split (train/test) from the raw common data
(items, first_activity, mas_trees, material_general, text embeddings).
No target values are used anywhere in this file -- all "train-fit" statistics
(frequency encodings, rare-category thresholds, PCA) are fit on items_train
only and then applied to both splits, so the output is safe to reuse across
all three challenges and across CV folds defined on top of it.

Run:
    python build_features.py

Outputs (next to this script):
    features_train.parquet, features_test.parquet, feature_list.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(r"D:\project\Sakko")
DATA = ROOT / "data"
FEAT = DATA / "train_test_features"
OUT = Path(__file__).parent

RARE_MIN_COUNT = 20  # raw train count below which a category is "rare"
KEY = ["Notification", "Item"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def read_items():
    tr = pd.read_csv(FEAT / "items_train.csv", low_memory=False)
    te = pd.read_csv(FEAT / "items_test.csv", low_memory=False)
    return tr, te


def freq_map(train_col: pd.Series):
    """Frequency (share of train rows) and raw count, fit on train only."""
    counts = train_col.value_counts(dropna=True)
    freqs = counts / len(train_col)
    return freqs, counts


def apply_freq(col: pd.Series, freqs: pd.Series):
    return col.map(freqs).fillna(0.0)


def apply_rare(col: pd.Series, counts: pd.Series, min_count=RARE_MIN_COUNT):
    mapped = col.map(counts)
    return ((mapped.isna()) | (mapped < min_count)).astype(int)


# --------------------------------------------------------------------------- #
# 1. temporal (10)
# --------------------------------------------------------------------------- #
def build_temporal(df: pd.DataFrame) -> pd.DataFrame:
    created_on = pd.to_datetime(df["Created on"].astype(str), format="%Y%m%d")
    hhmmss = df["Created At"].astype(str).str.zfill(6)
    hour = hhmmss.str[:2].astype(int)

    f = pd.DataFrame(index=df.index)
    f["year"] = created_on.dt.year
    f["month"] = created_on.dt.month
    f["quarter"] = created_on.dt.quarter
    f["day_of_week"] = created_on.dt.dayofweek
    f["hour"] = hour
    f["is_weekend"] = (created_on.dt.dayofweek >= 5).astype(int)
    f["month_sin"] = np.sin(2 * np.pi * f["month"] / 12)
    f["month_cos"] = np.cos(2 * np.pi * f["month"] / 12)
    f["dow_sin"] = np.sin(2 * np.pi * f["day_of_week"] / 7)
    f["dow_cos"] = np.cos(2 * np.pi * f["day_of_week"] / 7)
    return f, created_on


# --------------------------------------------------------------------------- #
# 2. defect group/code (16)
# --------------------------------------------------------------------------- #
DEFECT_COLS = {
    "problem_group": "Problem group text",
    "problem_code": "Problem Code Text",
    "obj_part_grp": "Obj. part grp. txt.",
    "obj_part_code": "Obj. part code text",
    "cause_group": "Cause group text",
    "cause_code": "Cause code text",
}


def fit_defect_stats(items_tr: pd.DataFrame):
    stats = {}
    for name, col in DEFECT_COLS.items():
        freqs, counts = freq_map(items_tr[col])
        stats[name] = (freqs, counts)
    # group -> distinct code count, fit on train
    ncodes = {
        "problem_group": items_tr.groupby(DEFECT_COLS["problem_group"])[DEFECT_COLS["problem_code"]].nunique(),
        "obj_part_grp": items_tr.groupby(DEFECT_COLS["obj_part_grp"])[DEFECT_COLS["obj_part_code"]].nunique(),
        "cause_group": items_tr.groupby(DEFECT_COLS["cause_group"])[DEFECT_COLS["cause_code"]].nunique(),
    }
    return stats, ncodes


def build_defect(df: pd.DataFrame, stats, ncodes) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for name, col in DEFECT_COLS.items():
        freqs, counts = stats[name]
        f[f"{name}_freq"] = apply_freq(df[col], freqs)
        f[f"{name}_rare"] = apply_rare(df[col], counts)
    f["problem_group_ncodes"] = df[DEFECT_COLS["problem_group"]].map(ncodes["problem_group"]).fillna(0)
    f["obj_part_grp_ncodes"] = df[DEFECT_COLS["obj_part_grp"]].map(ncodes["obj_part_grp"]).fillna(0)
    f["cause_group_ncodes"] = df[DEFECT_COLS["cause_group"]].map(ncodes["cause_group"]).fillna(0)
    f["defect_fields_any_missing"] = df[list(DEFECT_COLS.values())].isna().any(axis=1).astype(int)
    return f


# --------------------------------------------------------------------------- #
# 3. severity / priority / prevention (10)
# --------------------------------------------------------------------------- #
def fit_severity_stats(items_tr: pd.DataFrame):
    sev_freqs, sev_counts = freq_map(items_tr["Severity"])
    prio_freqs, prio_counts = freq_map(items_tr["Priority Text"])
    prev_freqs, prev_counts = freq_map(items_tr["Prevention Factor"])
    priority_cats = sorted(items_tr["Priority Text"].dropna().unique().tolist())
    return dict(
        sev=(sev_freqs, sev_counts),
        prio=(prio_freqs, prio_counts),
        prev=(prev_freqs, prev_counts),
        priority_cats=priority_cats,
    )


def build_severity(df: pd.DataFrame, s) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["severity_freq"] = apply_freq(df["Severity"], s["sev"][0])
    f["severity_rare"] = apply_rare(df["Severity"], s["sev"][1])
    f["priority_freq"] = apply_freq(df["Priority Text"], s["prio"][0])
    f["priority_missing"] = df["Priority Text"].isna().astype(int)
    for cat in s["priority_cats"]:
        f[f"priority_is_{cat}"] = (df["Priority Text"] == cat).astype(int)
    f["prevention_freq"] = apply_freq(df["Prevention Factor"], s["prev"][0])
    f["prevention_rare"] = apply_rare(df["Prevention Factor"], s["prev"][1])
    return f


# --------------------------------------------------------------------------- #
# 4. numeric quantities (8)
# --------------------------------------------------------------------------- #
def build_numeric(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    internal = df["DefectiveQty (Internal)"]
    external = df["DefectiveQty (External)"]
    f["number_of_defects_log1p"] = np.log1p(df["Number of Defects"])
    f["defective_qty_internal_log1p"] = np.log1p(internal)
    f["defective_qty_external_log1p"] = np.log1p(external)
    f["defective_total_log1p"] = np.log1p(internal + external)
    f["defective_internal_ratio"] = internal / (internal + external + 1e-6)
    f["complaint_reference_missing"] = df["Complaint Quantity"].isna().astype(int)
    f["complaint_quantity_log1p"] = np.log1p(df["Complaint Quantity"].fillna(0))
    f["reference_quantity_log1p"] = np.log1p(df["Reference Quantity"].fillna(0))
    return f


# --------------------------------------------------------------------------- #
# 5. material joins, Material role + Assembly role (16)
# --------------------------------------------------------------------------- #
def fit_material_stats(mg_train: pd.DataFrame):
    type_freqs, _ = freq_map(mg_train["Material_type"])
    group_freqs, _ = freq_map(mg_train["Material_Group"])
    return dict(type_freqs=type_freqs, group_freqs=group_freqs)


def build_material_role(df: pd.DataFrame, id_col: str, mg: pd.DataFrame, stats, role: str) -> pd.DataFrame:
    merged = df[[id_col]].merge(mg, left_on=id_col, right_on="Material", how="left")
    f = pd.DataFrame(index=df.index)
    f[f"{role}_gross_weight"] = merged["Gross_weight"].values
    f[f"{role}_net_weight"] = merged["Net_weight"].values
    f[f"{role}_volume"] = merged["Volume"].values
    f[f"{role}_material_type_freq"] = apply_freq(merged["Material_type"], stats["type_freqs"]).values
    f[f"{role}_material_group_freq"] = apply_freq(merged["Material_Group"], stats["group_freqs"]).values
    f[f"{role}_qm_procurement_known"] = merged["QM_Procurement"].notna().astype(int).values
    f[f"{role}_base_material_missing"] = merged["Base_Material"].isna().astype(int).values
    f[f"{role}_found"] = merged["Material_type"].notna().astype(int).values
    return f


# --------------------------------------------------------------------------- #
# 6. mas_trees aggregation, per split (8)
# --------------------------------------------------------------------------- #
def aggregate_trees(mt: pd.DataFrame) -> pd.DataFrame:
    g = mt.groupby("tree_id")
    agg = pd.DataFrame(
        {
            "tree_node_count": g.size(),
            "tree_max_depth": g["Depth"].max(),
            "tree_mean_depth": g["Depth"].mean(),
            "tree_in_mas_ratio": g["in_mas"].mean(),
            "tree_parent_order_missing_ratio": g["Parent Order"].apply(lambda s: s.isna().mean()),
            "tree_distinct_material_count": g["Material"].nunique(),
            "tree_distinct_parent_material_count": g["Parent Material"].nunique(),
        }
    )
    agg["tree_is_single_node"] = (agg["tree_node_count"] == 1).astype(int)
    return agg


def build_tree(df: pd.DataFrame, tree_agg: pd.DataFrame) -> pd.DataFrame:
    merged = df[["tree_id"]].merge(tree_agg, left_on="tree_id", right_index=True, how="left")
    return merged.drop(columns=["tree_id"])


# --------------------------------------------------------------------------- #
# 7. first activity (6)
# --------------------------------------------------------------------------- #
def fit_activity_stats(fa_train: pd.DataFrame):
    code_freqs, _ = freq_map(fa_train["Code Text for Activity"])
    group_freqs, _ = freq_map(fa_train["Code Group Text for Activity"])
    return dict(code_freqs=code_freqs, group_freqs=group_freqs)


def build_activity(df: pd.DataFrame, fa: pd.DataFrame, stats) -> pd.DataFrame:
    merged = df[KEY].merge(fa, on=KEY, how="left")
    f = pd.DataFrame(index=df.index)
    f["delay_h"] = merged["delay_h"].values
    f["delay_h_log1p"] = np.log1p(merged["delay_h"]).values
    f["has_first_activity"] = merged["delay_h"].notna().astype(int).values
    f["activity_code_freq"] = apply_freq(merged["Code Text for Activity"], stats["code_freqs"]).values
    f["activity_group_freq"] = apply_freq(merged["Code Group Text for Activity"], stats["group_freqs"]).values
    f["activity_seq_num"] = merged["activity"].str.extract(r"(\d+)").iloc[:, 0].astype(float).values
    return f


# --------------------------------------------------------------------------- #
# 8. notification-level structural stats (6)
# --------------------------------------------------------------------------- #
def build_notification(df: pd.DataFrame, created_on: pd.Series) -> pd.DataFrame:
    tmp = df[["Notification", "Created At", "Severity"]].copy()
    tmp["_created_on"] = created_on.values
    grp = tmp.groupby("Notification")

    items_per_notif = grp["Notification"].transform("size")
    order_rank = tmp.groupby("Notification")["Created At"].rank(method="first")
    date_span = (grp["_created_on"].transform("max") - grp["_created_on"].transform("min")).dt.days
    distinct_sev = grp["Severity"].transform("nunique")

    f = pd.DataFrame(index=df.index)
    f["items_per_notification"] = items_per_notif.values
    f["item_position_in_notification"] = order_rank.values
    f["item_position_frac"] = (order_rank / items_per_notif).values
    f["is_only_item_in_notification"] = (items_per_notif == 1).astype(int).values
    f["notification_date_span_days"] = date_span.values
    f["notification_distinct_severity_count"] = distinct_sev.values
    return f


# --------------------------------------------------------------------------- #
# 9. text embeddings -> PCA (item_longtext 8, first_activity 4, material_desc 4)
# --------------------------------------------------------------------------- #
def fit_pca_family(fam: str, n_components: int, sample_cap: int = 30000):
    keys_tr = pd.read_parquet(FEAT / f"{fam}_train_keys.parquet")
    arr_tr = np.load(FEAT / f"{fam}_train_embeddings.npy", mmap_mode="r")
    n = arr_tr.shape[0]
    if n > sample_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=sample_cap, replace=False)
        idx.sort()
        sample = arr_tr[idx]
    else:
        sample = arr_tr[:]
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(np.asarray(sample, dtype=np.float32))
    return pca, keys_tr


def build_text_pca(df: pd.DataFrame, fam: str, split: str, pca: PCA, prefix: str, item_join_col=None) -> pd.DataFrame:
    """item_join_col=None joins on (Notification, Item); otherwise joins on that single column (e.g. 'Material')."""
    keys = pd.read_parquet(FEAT / f"{fam}_{split}_keys.parquet")
    arr = np.load(FEAT / f"{fam}_{split}_embeddings.npy", mmap_mode="r")
    comps = pca.transform(np.asarray(arr, dtype=np.float32))
    comp_cols = [f"{prefix}_pca{i}" for i in range(comps.shape[1])]
    comp_df = pd.DataFrame(comps, columns=comp_cols)

    if item_join_col is None:
        comp_df["Notification"] = keys["notif"].values
        comp_df["Item"] = keys["item"].values
        comp_df = comp_df.drop_duplicates(subset=["Notification", "Item"])
        merged = df[["Notification", "Item"]].merge(comp_df, on=["Notification", "Item"], how="left")
    else:
        comp_df[item_join_col] = keys[item_join_col].values
        comp_df = comp_df.drop_duplicates(subset=[item_join_col])
        merged = df[[item_join_col]].merge(comp_df, on=item_join_col, how="left")
    return merged[comp_cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    items_tr, items_te = read_items()

    parts_tr, parts_te = [], []

    temp_tr, created_on_tr = build_temporal(items_tr)
    temp_te, created_on_te = build_temporal(items_te)
    parts_tr.append(temp_tr)
    parts_te.append(temp_te)

    defect_stats, ncodes = fit_defect_stats(items_tr)
    parts_tr.append(build_defect(items_tr, defect_stats, ncodes))
    parts_te.append(build_defect(items_te, defect_stats, ncodes))

    sev_stats = fit_severity_stats(items_tr)
    parts_tr.append(build_severity(items_tr, sev_stats))
    parts_te.append(build_severity(items_te, sev_stats))

    parts_tr.append(build_numeric(items_tr))
    parts_te.append(build_numeric(items_te))

    mg_tr = pd.read_csv(FEAT / "material_general_train.csv")
    mg_te = pd.read_csv(FEAT / "material_general_test.csv")
    mat_stats = fit_material_stats(mg_tr)
    parts_tr.append(build_material_role(items_tr, "Material", mg_tr, mat_stats, "material"))
    parts_tr.append(build_material_role(items_tr, "Assembly", mg_tr, mat_stats, "assembly"))
    parts_te.append(build_material_role(items_te, "Material", mg_te, mat_stats, "material"))
    parts_te.append(build_material_role(items_te, "Assembly", mg_te, mat_stats, "assembly"))

    mt_tr = pd.read_csv(FEAT / "mas_trees_train.csv")
    mt_te = pd.read_csv(FEAT / "mas_trees_test.csv")
    tree_agg_tr = aggregate_trees(mt_tr)
    tree_agg_te = aggregate_trees(mt_te)
    parts_tr.append(build_tree(items_tr, tree_agg_tr))
    parts_te.append(build_tree(items_te, tree_agg_te))

    fa_tr = pd.read_csv(FEAT / "first_activity_train.csv")
    fa_te = pd.read_csv(FEAT / "first_activity_test.csv")
    act_stats = fit_activity_stats(fa_tr)
    parts_tr.append(build_activity(items_tr, fa_tr, act_stats))
    parts_te.append(build_activity(items_te, fa_te, act_stats))

    parts_tr.append(build_notification(items_tr, created_on_tr))
    parts_te.append(build_notification(items_te, created_on_te))

    print("fitting PCA: item_longtext (8) ...")
    pca_long, _ = fit_pca_family("item_longtext", 8)
    parts_tr.append(build_text_pca(items_tr, "item_longtext", "train", pca_long, "item_longtext"))
    parts_te.append(build_text_pca(items_te, "item_longtext", "test", pca_long, "item_longtext"))

    print("fitting PCA: first_activity text (4) ...")
    pca_act, _ = fit_pca_family("first_activity", 4)
    parts_tr.append(build_text_pca(items_tr, "first_activity", "train", pca_act, "activity_text"))
    parts_te.append(build_text_pca(items_te, "first_activity", "test", pca_act, "activity_text"))

    print("fitting PCA: material_desc (4) ...")
    pca_mat, _ = fit_pca_family("material_desc", 4)
    parts_tr.append(build_text_pca(items_tr, "material_desc", "train", pca_mat, "material_desc", item_join_col="Material"))
    parts_te.append(build_text_pca(items_te, "material_desc", "test", pca_mat, "material_desc", item_join_col="Material"))

    feats_tr = pd.concat(parts_tr, axis=1)
    feats_te = pd.concat(parts_te, axis=1)

    feats_tr.insert(0, "Item", items_tr["Item"].values)
    feats_tr.insert(0, "Notification", items_tr["Notification"].values)
    feats_te.insert(0, "Item", items_te["Item"].values)
    feats_te.insert(0, "Notification", items_te["Notification"].values)

    feats_tr.to_parquet(OUT / "features_train.parquet", index=False)
    feats_te.to_parquet(OUT / "features_test.parquet", index=False)

    feature_cols = [c for c in feats_tr.columns if c not in KEY]
    (OUT / "feature_list.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    print(f"train shape {feats_tr.shape}, test shape {feats_te.shape}, n_features={len(feature_cols)}")


if __name__ == "__main__":
    main()
