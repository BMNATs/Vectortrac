"""
เปรียบเทียบ K-Means (baseline) กับ Hierarchical (Ward) และ GMM ด้วย silhouette และ elbow (WCSS)

ไฟล์นี้แยกจาก dengue_clustering.py ไม่แก้ไฟล์เดิม
วิธีใช้:
  1. รัน dengue_clustering.py ตามปกติก่อน เพื่อให้ได้ dengue_clustering_result.csv
  2. วางไฟล์นี้ไว้โฟลเดอร์เดียวกัน แล้วรัน: python compare_clustering.py
ผลลัพธ์: ตารางพิมพ์ในหน้าจอ + dengue_model_comparison.png (กราฟ) + dengue_model_labels.csv (กลุ่มของแต่ละอำเภอจากทั้ง 3 โมเดล)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIG (ให้ตรงกับ dengue_clustering.py)
# =====================================================================
INPUT_FILE = 'dengue_clustering_result.csv'
PLOT_FILE = 'dengue_model_comparison.png'
LABELS_FILE = 'dengue_model_labels.csv'

K = 6                          # จำนวนกลุ่มที่ใช้จริง
K_RANGE = range(2, 9)          # ช่วง k ที่ทดลอง
FEATURES = ['log_incidence', 'is_metro', 'is_rainy', 'is_youth', 'endemic_years']
MODELS = ['K-Means', 'Hierarchical (Ward)', 'GMM']     # K-Means = baseline


# =====================================================================
# Helpers
# =====================================================================
def fit_labels(name, X, k, seed=42):
    if name == 'K-Means':
        return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(X)
    if name == 'Hierarchical (Ward)':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
    if name == 'GMM':
        # covariance แบบ diag เพราะมีแค่ ~66 อำเภอ 5 feature (แบบ full ประมาณพารามิเตอร์ไม่ไหว)
        return GaussianMixture(n_components=k, covariance_type='diag',
                               n_init=10, random_state=seed).fit(X).predict(X)
    raise ValueError(name)


def wcss(X, labels):
    """Within-cluster sum of squares ใช้วาด elbow คิดจาก label ด้วยสูตรเดียวกันทุกโมเดล"""
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        total += ((pts - pts.mean(axis=0)) ** 2).sum()
    return float(total)


def reorder_by_incidence(labels, incidence):
    """เรียงเลขกลุ่มใหม่ 0 = อัตราป่วยเฉลี่ยต่ำสุด ... K-1 = สูงสุด (เหมือน dengue_clustering.py)"""
    s = pd.DataFrame({'c': labels, 'inc': np.asarray(incidence)})
    order = s.groupby('c')['inc'].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[c] for c in labels])


# =====================================================================
# Main
# =====================================================================
def main():
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    except FileNotFoundError:
        raise SystemExit(f'ไม่พบ {INPUT_FILE} รัน dengue_clustering.py ก่อน แล้ววางไฟล์นี้ไว้โฟลเดอร์เดียวกัน')

    need = ['province', 'district', 'cluster', 'incidence_rate', 'is_metro', 'is_rainy',
            'is_youth', 'endemic_years']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f'{INPUT_FILE} ไม่มีคอลัมน์ {missing}')

    train = df[df['cluster'] >= 0].copy()             # -1 = ข้อมูลไม่เพียงพอ ไม่เข้าโมเดลเหมือนไฟล์เดิม
    if len(train) <= K:
        raise SystemExit(f'อำเภอที่เข้าโมเดลมี {len(train)} แห่ง น้อยเกินไปสำหรับ k={K}')
    train['log_incidence'] = np.log1p(train['incidence_rate'])

    X = train[FEATURES]
    X = X.fillna(X.median())
    X = StandardScaler().fit_transform(X)
    inc = train['incidence_rate'].values
    print(f'อ่าน {INPUT_FILE}: {len(train)} อำเภอที่เข้าโมเดล, features = {FEATURES}')

    # ตรวจว่า feature ที่สร้างใหม่ตรงกับของเดิม: K-Means ที่รันที่นี่ต้องได้กลุ่มเดียวกับในไฟล์ CSV
    km = reorder_by_incidence(fit_labels('K-Means', X, K), inc)
    same = (km == train['cluster'].values).mean() * 100
    print(f'ตรวจความถูกต้อง: K-Means (k={K}) ที่นี่ตรงกับคอลัมน์ cluster ใน CSV {same:.1f}% ของอำเภอ')
    if same < 100:
        print('   (ไม่ถึง 100% ได้ถ้าบางอำเภอมีค่า feature ว่างที่ไฟล์เดิมเติม 0 ก่อนบันทึก ต่างกันเล็กน้อยไม่กระทบการเปรียบเทียบ)')

    # ---- silhouette + elbow ----
    ks = [k for k in K_RANGE if k < len(X)]
    sil = {m: [] for m in MODELS}
    inertia = {m: [] for m in MODELS}
    for m in MODELS:
        for k in ks:
            lab = fit_labels(m, X, k)
            sil[m].append(silhouette_score(X, lab) if len(set(lab)) > 1 else np.nan)
            inertia[m].append(wcss(X, lab))
    idx = [f'k={k}' for k in ks]
    sil_df = pd.DataFrame(sil, index=idx)
    wcss_df = pd.DataFrame(inertia, index=idx)

    print('\nSilhouette score (ยิ่งสูงยิ่งแยกกลุ่มชัด):')
    print(sil_df.round(3).to_string())
    print('\nElbow: WCSS (ยิ่งต่ำยิ่งกระชับ ดูจุดที่ค่าเริ่มลดช้าลงมาก ไม่ใช่จุดต่ำสุด):')
    print(wcss_df.round(1).to_string())

    print('\nk ที่ silhouette สูงสุดของแต่ละโมเดล:')
    for m in MODELS:
        j = int(np.nanargmax(sil_df[m].values))
        print(f'  {m}: k={ks[j]} ({sil_df[m].iloc[j]:.3f})')

    if K in ks:
        j = ks.index(K)
        print(f'\nเทียบที่ k={K} (ที่ใช้จริง):')
        print(pd.DataFrame({'silhouette': [sil[m][j] for m in MODELS],
                            'WCSS': [inertia[m][j] for m in MODELS]}, index=MODELS).round(3).to_string())

    # ---- ขนาดกลุ่มที่ k=K (เลขกลุ่มเรียงตามอัตราป่วยแล้ว) ----
    labels_out = train[['province', 'district']].copy()
    labels_out['incidence_rate'] = train['incidence_rate'].round(1)
    sizes = {}
    for m in MODELS:
        lab = reorder_by_incidence(fit_labels(m, X, K), inc)
        labels_out[m] = lab
        sizes[m] = pd.Series(lab).value_counts().reindex(range(K), fill_value=0)
    sizes_df = pd.DataFrame(sizes)
    sizes_df.index.name = 'cluster'
    print(f'\nจำนวนอำเภอในแต่ละกลุ่มที่ k={K} (0 = ป่วยน้อยสุด):')
    print(sizes_df.to_string())
    tiny = {m: int(sizes_df[m].min()) for m in MODELS if sizes_df[m].min() < 3}
    if tiny:
        print(f'⚠ กลุ่มเล็กมาก (น้อยกว่า 3 อำเภอ): {tiny} โมเดลนี้อาจแบ่งกลุ่มได้ไม่เสถียร')

    labels_out.to_csv(LABELS_FILE, index=False, encoding='utf-8-sig')
    print(f"\nบันทึกกลุ่มของแต่ละอำเภอจากทั้ง 3 โมเดลที่ '{LABELS_FILE}'")

    # ---- กราฟ ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for m in MODELS:
            axes[0].plot(ks, inertia[m], marker='o', label=m)
            axes[1].plot(ks, sil[m], marker='o', label=m)
        axes[0].set_title('Elbow method (WCSS, lower = tighter)')
        axes[0].set_ylabel('WCSS')
        axes[1].set_title('Silhouette score (higher = better separated)')
        axes[1].set_ylabel('Silhouette')
        for ax in axes:
            ax.axvline(K, color='gray', linestyle='--', linewidth=1, label=f'k used = {K}')
            ax.set_xlabel('Number of clusters (k)')
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_FILE, dpi=150)
        plt.close(fig)
        print(f"บันทึกกราฟที่ '{PLOT_FILE}'")
    except ImportError:
        print('(ไม่ได้ติดตั้ง matplotlib จึงไม่วาดกราฟ ติดตั้งด้วย: pip install matplotlib)')


if __name__ == '__main__':
    main()