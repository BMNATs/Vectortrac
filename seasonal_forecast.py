"""
รูปแบบตามฤดูกาลของไข้เลือดออก + ทดสอบย้อนหลังว่า "ทำนายได้ดีกว่าวิธีง่ายๆ ไหม"
แยกเป็นไฟล์ต่างหาก ไม่แก้ dengue_clustering.py (แค่เรียกฟังก์ชันอ่านข้อมูลจากไฟล์นั้น)

วิธีรัน: วางไฟล์นี้ไว้โฟลเดอร์เดียวกับ dengue_clustering.py และ Dengue_odpc4_*.csv แล้วรัน
    python seasonal_forecast.py

ผลลัพธ์:
    dengue_seasonal_profile.csv   สัดส่วนผู้ป่วยรายเดือน (ค่าเฉลี่ย ต่ำสุด สูงสุดระหว่างปี) รายจังหวัด และรายกลุ่มคลัสเตอร์
    dengue_seasonal_backtest.csv  ผลทดสอบย้อนหลังรายจังหวัด
    dengue_seasonal.png           กราฟ
"""
import os
import warnings

import numpy as np
import pandas as pd

from dengue_clustering import OUTPUT_FILE, load_dengue, norm_name

warnings.filterwarnings('ignore')

PROFILE_FILE = 'dengue_seasonal_profile.csv'
BACKTEST_FILE = 'dengue_seasonal_backtest.csv'
PLOT_FILE = 'dengue_seasonal.png'

MIN_YEAR_CASES = 30   # ปี x จังหวัด ที่มีเคสที่อ่านเดือนได้น้อยกว่านี้ ไม่นำมาสร้างรูปแบบ (น้อยเกินไปจะแกว่งตามโอกาส)
MONTH_TH = ['ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.']


# ---------------------------------------------------------------------
# 1) ตารางเคสรายเดือน
# ---------------------------------------------------------------------
def monthly_counts(dengue, by):
    """คืน DataFrame index=(by..., year_be) คอลัมน์ 1..12 = จำนวนเคส (เฉพาะแถวที่อ่านเดือนได้)"""
    d = dengue.dropna(subset=['month']).copy()
    d['month'] = d['month'].astype(int)
    t = d.groupby(by + ['year_be', 'month']).size().unstack('month', fill_value=0)
    return t.reindex(columns=range(1, 13), fill_value=0)


def to_share(t):
    tot = t.sum(axis=1)
    keep = tot >= MIN_YEAR_CASES
    return t[keep].div(tot[keep], axis=0), tot[keep]


def profile_table(share, level_names):
    """เฉลี่ยสัดส่วนรายเดือนข้ามปี พร้อมต่ำสุด/สูงสุดของแต่ละเดือน"""
    rows = []
    for key, g in share.groupby(level=list(range(len(level_names)))):
        key = key if isinstance(key, tuple) else (key,)
        mean, lo, hi = g.mean(), g.min(), g.max()
        peak = int(mean.idxmax())
        for m in range(1, 13):
            rows.append({**dict(zip(level_names, key)), 'month': m, 'month_th': MONTH_TH[m - 1],
                         'share_mean': mean[m], 'share_min': lo[m], 'share_max': hi[m],
                         'n_years': len(g), 'is_peak': m == peak})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2) ทดสอบย้อนหลัง (leave-one-year-out) ระดับจังหวัด
# ---------------------------------------------------------------------
def backtest(cnt):
    """
    ทดสอบ 2 งาน โดยเว้นปีที่จะทำนายออก แล้วสร้างรูปแบบจากปีที่เหลือ
      งาน A  ทำนายสัดส่วนผู้ป่วยรายเดือนของปีที่เว้นไว้  (วัดด้วย MAE ของสัดส่วน และตรวจว่าเดือนที่สูงสุดคลาดไม่เกิน 1 เดือน)
             - ฤดูกาล   = ค่าเฉลี่ยสัดส่วนของปีอื่น
             - วิธีง่าย = ทุกเดือนเท่ากัน (1/12)
      งาน B  ทำนายจำนวนเคส "เดือนถัดไป" จากเดือนนี้  (วัดด้วย MAE ของจำนวนเคส)
             - ฤดูกาล   = เคสเดือนนี้ x (สัดส่วนปกติของเดือนหน้า / สัดส่วนปกติของเดือนนี้)
             - วิธีง่าย = เท่ากับเดือนนี้ (persistence)
    """
    rows = []
    for prov, g in cnt.groupby(level=0):
        g = g.droplevel(0)
        share, tot = to_share(g)
        years = list(share.index)
        if len(years) < 3:
            continue
        a_season, a_flat, peak_hit = [], [], []
        b_season, b_naive = [], []
        for y in years:
            others = [o for o in years if o != y]
            prof = share.loc[others].mean()
            actual = share.loc[y]
            a_season.append((prof - actual).abs().mean())
            a_flat.append((1 / 12 - actual).abs().mean())
            diff = abs(int(prof.idxmax()) - int(actual.idxmax()))
            peak_hit.append(min(diff, 12 - diff) <= 1)

            c = g.loc[y]
            for m in range(1, 12):
                if c[m] <= 0 or prof[m] <= 0:
                    continue
                pred_s = c[m] * prof[m + 1] / prof[m]
                b_season.append(abs(pred_s - c[m + 1]))
                b_naive.append(abs(c[m] - c[m + 1]))
        rows.append({
            'province': prov, 'n_years': len(years), 'years': ','.join(str(int(y)) for y in years),
            'A_mae_season': np.mean(a_season), 'A_mae_flat': np.mean(a_flat),
            'A_peak_within_1_month': float(np.mean(peak_hit)),
            'B_mae_season': np.mean(b_season) if b_season else np.nan,
            'B_mae_persistence': np.mean(b_naive) if b_naive else np.nan,
        })
    out = pd.DataFrame(rows)
    out['A_season_wins'] = out['A_mae_season'] < out['A_mae_flat']
    out['B_season_wins'] = out['B_mae_season'] < out['B_mae_persistence']
    return out


# ---------------------------------------------------------------------
# 3) กราฟ
# ---------------------------------------------------------------------
def draw(prov_prof, clu_prof, bt):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('(ไม่ได้ติดตั้ง matplotlib จึงไม่วาดกราฟ ติดตั้งด้วย: pip install matplotlib)')
        return
    plt.rcParams['font.family'] = ['Tahoma', 'Thonburi', 'Sarabun', 'Noto Sans Thai', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    provs = list(prov_prof['province'].unique())
    n = len(provs) + (1 if clu_prof is not None else 0)
    cols = 3
    rows = int(np.ceil(n / cols)) + 1
    fig = plt.figure(figsize=(15, 3.4 * rows))
    gs = fig.add_gridspec(rows, cols)
    x = np.arange(1, 13)

    for i, p in enumerate(provs):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        d = prov_prof[prov_prof['province'] == p].sort_values('month')
        ax.fill_between(x, d['share_min'] * 100, d['share_max'] * 100, color='#9ecae1', alpha=.6, label='ต่ำสุด-สูงสุดระหว่างปี')
        ax.plot(x, d['share_mean'] * 100, color='#08519c', lw=2, marker='o', ms=3, label='ค่าเฉลี่ย')
        ax.axhline(100 / 12, color='grey', ls=':', lw=1)
        ax.set_title(f'{p} (ใช้ {int(d["n_years"].iloc[0])} ปี)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_TH, fontsize=7, rotation=60)
        ax.set_ylabel('% ของผู้ป่วยทั้งปี', fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)

    if clu_prof is not None:
        k = len(provs)
        ax = fig.add_subplot(gs[k // cols, k % cols])
        for c, d in clu_prof.groupby('cluster'):
            d = d.sort_values('month')
            ax.plot(x, d['share_mean'] * 100, marker='o', ms=3, label=f'กลุ่ม {c}')
        ax.axhline(100 / 12, color='grey', ls=':', lw=1)
        ax.set_title('แยกตามกลุ่มคลัสเตอร์ (ค่าเฉลี่ย)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_TH, fontsize=7, rotation=60)
        ax.legend(fontsize=7, ncol=2)

    if len(bt):
        ax = fig.add_subplot(gs[rows - 1, 0:2])
        xs = np.arange(len(bt))
        ax.bar(xs - .2, bt['A_mae_flat'] * 100, .4, color='#bdbdbd', label='วิธีง่าย (ทุกเดือนเท่ากัน)')
        ax.bar(xs + .2, bt['A_mae_season'] * 100, .4, color='#08519c', label='รูปแบบตามฤดูกาล')
        ax.set_xticks(xs)
        ax.set_xticklabels(bt['province'], fontsize=8, rotation=20)
        ax.set_ylabel('ความคลาดเคลื่อนเฉลี่ย (จุด % ต่อเดือน)')
        ax.set_title('ทดสอบย้อนหลัง งาน A: ทำนายสัดส่วนรายเดือนของปีที่เว้นไว้ (ต่ำ = ดี)', fontsize=10)
        ax.legend(fontsize=8)
        ax = fig.add_subplot(gs[rows - 1, 2])
        ax.bar(xs - .2, bt['B_mae_persistence'], .4, color='#bdbdbd', label='เท่ากับเดือนนี้')
        ax.bar(xs + .2, bt['B_mae_season'], .4, color='#08519c', label='ปรับตามฤดูกาล')
        ax.set_xticks(xs)
        ax.set_xticklabels(bt['province'], fontsize=7, rotation=60)
        ax.set_ylabel('MAE (ราย)')
        ax.set_title('งาน B: ทำนายเคสเดือนถัดไป (ต่ำ = ดี)', fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130)
    print(f'บันทึกกราฟที่ {PLOT_FILE}')


# ---------------------------------------------------------------------
def main():
    print('1. อ่านข้อมูลผู้ป่วย (ใช้ฟังก์ชันเดียวกับ dengue_clustering.py)...')
    dengue = load_dengue()
    ok = dengue.dropna(subset=['month'])
    years = sorted(ok['year_be'].unique())
    print(f'   ปีที่อ่านเดือนได้: {[int(y) for y in years]}  รวม {len(ok):,} จาก {len(dengue):,} เคส')
    print(f'   ปีที่ถูกตัดออกเพราะวันที่ใช้ไม่ได้: {sorted(int(v) for v in set(dengue["year_be"].unique()) - set(years))}')

    print('\n2. สร้างรูปแบบตามฤดูกาลรายจังหวัด...')
    cnt = monthly_counts(dengue, ['province'])
    share, tot = to_share(cnt)
    dropped = cnt.index.difference(share.index)
    if len(dropped):
        print(f'   ไม่ใช้ (เคสน้อยกว่า {MIN_YEAR_CASES} ในปีนั้น): {list(dropped)}')
    prov_prof = profile_table(share, ['province'])
    for p, d in prov_prof.groupby('province'):
        top = d.sort_values('share_mean', ascending=False).head(3)
        txt = ', '.join(f'{r.month_th} {r.share_mean:.0%}' for r in top.itertuples())
        print(f'   {p:<16} เดือนที่สูงสุด 3 อันดับ: {txt}  (ใช้ {int(d["n_years"].iloc[0])} ปี)')

    clu_prof = None
    if os.path.exists(OUTPUT_FILE):
        print('\n3. สร้างรูปแบบตามฤดูกาลรายกลุ่มคลัสเตอร์...')
        res = pd.read_csv(OUTPUT_FILE)
        res = res[res['cluster'] >= 0]
        res['key'] = res['district'].map(norm_name)
        d2 = dengue.copy()
        d2['key'] = d2['district'].map(norm_name)
        d2 = d2.merge(res[['province', 'key', 'cluster']], on=['province', 'key'], how='inner')
        print(f'   จับคู่ได้ {len(d2):,} เคส จาก {len(dengue):,}')
        if len(d2):
            cc = monthly_counts(d2, ['cluster'])
            cs, _ = to_share(cc)
            clu_prof = profile_table(cs, ['cluster'])
            for c, d in clu_prof.groupby('cluster'):
                pk = d.sort_values('share_mean', ascending=False).iloc[0]
                print(f'   กลุ่ม {c}: เดือนสูงสุด {pk.month_th} ({pk.share_mean:.0%} ของผู้ป่วยทั้งปี)')

    out = prov_prof.assign(level='province').rename(columns={'province': 'name'})
    if clu_prof is not None:
        out = pd.concat([out, clu_prof.assign(level='cluster').rename(columns={'cluster': 'name'})], ignore_index=True)
    out.to_csv(PROFILE_FILE, index=False, encoding='utf-8-sig')

    print('\n4. ทดสอบย้อนหลัง (เว้นปีที่จะทำนายออกทีละปี)...')
    bt = backtest(cnt)
    bt.to_csv(BACKTEST_FILE, index=False, encoding='utf-8-sig')
    if len(bt):
        show = bt[['province', 'n_years', 'A_mae_season', 'A_mae_flat', 'A_peak_within_1_month',
                   'B_mae_season', 'B_mae_persistence']].copy()
        show[['A_mae_season', 'A_mae_flat']] *= 100
        print(show.round(2).to_string(index=False))
        print('\nสรุป')
        print(f'  งาน A (ทำนายสัดส่วนรายเดือน): รูปแบบฤดูกาลชนะวิธีง่าย {int(bt.A_season_wins.sum())} จาก {len(bt)} จังหวัด')
        print(f'  งาน A: เดือนที่สูงสุดทำนายคลาดไม่เกิน 1 เดือน เฉลี่ย {bt.A_peak_within_1_month.mean():.0%}')
        print(f'  งาน B (ทำนายเคสเดือนถัดไป): ปรับตามฤดูกาลชนะ "เท่ากับเดือนนี้" {int(bt.B_season_wins.sum())} จาก {len(bt)} จังหวัด')
        print('  ข้อควรระวัง: ทดสอบจากแค่', len(years), 'ปี ปีที่ผิดปกติ (เช่น 2564 ช่วงโควิด) มีผลมาก ใช้เป็นแนวทางเตรียมตัว ไม่ใช่การพยากรณ์จำนวนผู้ป่วย')
    else:
        print('   ข้อมูลไม่พอสำหรับทดสอบ (ต้องมีอย่างน้อย 3 ปีต่อจังหวัด)')

    draw(prov_prof, clu_prof, bt)
    print(f'\nบันทึกแล้ว: {PROFILE_FILE}, {BACKTEST_FILE}')


if __name__ == '__main__':
    main()