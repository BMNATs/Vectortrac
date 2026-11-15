import difflib
import glob
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIG
# =====================================================================
OUTPUT_FILE = 'dengue_clustering_result.csv'
UNMATCHED_FILE = 'dengue_unmatched_districts.csv'   # อำเภอที่ merge ประชากรไม่ติด (ไว้ตรวจสอบ)
POP_FILE = 'stat_t66.xls'   # ไฟล์ประชากรระดับตำบล/หมู่บ้าน ปี 2566

K = 6                       # จำนวนกลุ่ม (ดูตาราง silhouette ที่พิมพ์ออกมาประกอบการตัดสินใจ)
MIN_CASES = 20              # อำเภอที่มีเคสรวมน้อยกว่านี้ จะไม่เข้าโมเดล (cluster = -1)
ENDEMIC_CASES = 20          # ปีที่มีเคส >= ค่านี้ นับเป็น "ปีที่ระบาด"
RAINY_MONTHS = [6, 7, 8, 9, 10]
YOUTH_MAX_AGE = 24
TARGET_PROVINCES = None     # เช่น ['นนทบุรี', 'ปทุมธานี'] ; None = ใช้ทุกจังหวัดที่มีในข้อมูล

# ถ้าชื่ออำเภอไม่ตรงกับไฟล์ประชากร ให้ใส่ชื่อที่ถูกต้องที่นี่ (ดูคำแนะนำที่สคริปต์พิมพ์ในขั้นตอนที่ 4)
# รูปแบบ: {('จังหวัด', 'ชื่อในไฟล์ผู้ป่วย'): 'ชื่อในไฟล์ประชากร'}
DISTRICT_ALIASES = {}

# รหัสอำเภอ 4 หลัก (จังหวัด2 + อำเภอ2 เช่น 1303) -> ชื่ออำเภอ ใช้เมื่อสคริปต์หาชื่อจากไฟล์ไม่ได้
# 1303 = ธัญบุรี: ในไฟล์ประชากรมีรหัส 1301,1302,1304-1307 ขาดเฉพาะ 1303 ตามลำดับอำเภอของปทุมธานี
DISTRICT_CODE_NAMES = {1303: 'ธัญบุรี'}

# ถ้าเดือนเดียวมีเคสเกินสัดส่วนนี้ของทั้งไฟล์ (เช่น ทุกแถวเป็น 1 ม.ค.) ถือว่าวันที่ป่วยเป็นค่าสมมติ
# จะไม่ใช้ไฟล์นั้นคำนวณ is_rainy (แต่ยังนับจำนวนเคสตามปกติ)
MAX_SINGLE_MONTH_SHARE = 0.5

FEATURES = ['log_incidence', 'is_metro', 'is_rainy', 'is_youth', 'endemic_years']

THAI_MONTHS = {
    'ม.ค.': 1, 'มกราคม': 1, 'ก.พ.': 2, 'กุมภาพันธ์': 2, 'มี.ค.': 3, 'มีนาคม': 3,
    'เม.ย.': 4, 'เมษายน': 4, 'พ.ค.': 5, 'พฤษภาคม': 5, 'มิ.ย.': 6, 'มิถุนายน': 6,
    'ก.ค.': 7, 'กรกฎาคม': 7, 'ส.ค.': 8, 'สิงหาคม': 8, 'ก.ย.': 9, 'กันยายน': 9,
    'ต.ค.': 10, 'ตุลาคม': 10, 'พ.ย.': 11, 'พฤศจิกายน': 11, 'ธ.ค.': 12, 'ธันวาคม': 12,
}
ENG_MONTHS = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}


# =====================================================================
# Helpers
# =====================================================================
def find_year(filename):
    """ดึงปี พ.ศ. 2561-2566 จากชื่อไฟล์ ถ้าไม่เจอคืน None"""
    m = re.search(r'256[1-6]', filename)
    return int(m.group()) if m else None


def read_csv_safe(path):
    for enc in ('utf-8', 'utf-8-sig', 'cp874'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'อ่านไฟล์ {path} ไม่ได้ (ลอง utf-8 / utf-8-sig / cp874 แล้ว)')


def detect_date_order(series):
    """
    ตรวจรูปแบบวันที่ของทั้งไฟล์: 'dmy' (DD/MM/YYYY) หรือ 'mdy' (MM/DD/YYYY)
    ถ้าตัวเลขตำแหน่งที่ 2 เคยเกิน 12 แต่ตำแหน่งแรกไม่เกิน 12 แสดงว่าเป็น MM/DD
    """
    first, second = [], []
    for s in series.dropna().astype(str):
        nums = re.findall(r'\d+', s)
        if len(nums) >= 3 and len(nums[0]) <= 2 and len(nums[1]) <= 2:
            first.append(int(nums[0]))
            second.append(int(nums[1]))
    if not first:
        return 'dmy'
    if max(second) > 12 and max(first) <= 12:
        return 'mdy'
    return 'dmy'


def extract_month(value, order='dmy'):
    """สกัดเดือน (1-12) จากข้อความวันที่ ถ้าอ่านไม่ได้คืน NaN (ไม่ใช่ 0)"""
    s = str(value).strip().lower()
    if s in ('', 'nan', 'nat', 'none', '00/00/0000', '0000-00-00'):
        return np.nan

    for k, v in THAI_MONTHS.items():
        if k in s:
            return v
    for k, v in ENG_MONTHS.items():
        if k in s:
            return v

    nums = re.findall(r'\d+', s)
    if len(nums) < 2:
        return np.nan

    # YYYY-MM-DD
    if len(nums[0]) == 4:
        m = int(nums[1])
        return m if 1 <= m <= 12 else np.nan

    # DD/MM/YYYY หรือ MM/DD/YYYY
    if len(nums) >= 3:
        a, b = int(nums[0]), int(nums[1])
        if a > 12 and b <= 12:
            m = b                              # a เป็นวัน
        elif b > 12 and a <= 12:
            m = a                              # b เป็นวัน
        else:
            m = a if order == 'mdy' else b     # กำกวม -> ใช้รูปแบบที่ตรวจเจอของไฟล์นั้น
        return m if 1 <= m <= 12 else np.nan

    return np.nan


def norm_name(s):
    """ทำชื่ออำเภอให้เทียบกันได้ (ตัดคำนำหน้า/ช่องว่าง)"""
    s = str(s)
    for p in ('ท้องถิ่นอำเภอ', 'ท้องถิ่นเขต', 'อำเภอ'):
        s = s.replace(p, '')
    s = re.sub(r'^อ\.', '', s.strip())
    s = re.sub(r'\s+', '', s)
    return re.sub(r'^เมือง(?=.)', '', s)   # 'เมืองX' กับ 'X' ถือเป็นอำเภอเดียวกัน


def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False), errors='coerce')


# =====================================================================
# Load data
# =====================================================================
def load_dengue():
    files = [f for f in sorted(glob.glob('*.csv'))
             if f not in (OUTPUT_FILE, UNMATCHED_FILE) and find_year(f)]
    if not files:
        raise SystemExit('ไม่พบไฟล์ CSV ผู้ป่วย (ชื่อไฟล์ต้องมีปี 2561-2566) ในโฟลเดอร์นี้')

    dfs = []
    for f in files:
        df = read_csv_safe(f)
        df.columns = [c.lower() for c in df.columns]
        df['year_be'] = find_year(f)

        if 'datesick' in df.columns:
            order = detect_date_order(df['datesick'])
            df['month'] = df['datesick'].apply(lambda v: extract_month(v, order))
        else:
            order = '-'
            df['month'] = np.nan

        bad = df['month'].isna().mean() * 100
        note = ''
        counts = df['month'].dropna().value_counts(normalize=True)
        if len(counts) and counts.iloc[0] > MAX_SINGLE_MONTH_SHARE:
            note = (f' | ⚠ เดือน {int(counts.index[0])} มีเคส {counts.iloc[0]:.0%} ของไฟล์'
                    ' -> วันที่น่าจะเป็นค่าสมมติ ไม่ใช้ไฟล์นี้คำนวณ is_rainy')
            df['month'] = np.nan
        print(f'   {f}: {len(df):,} แถว | รูปแบบวันที่={order} | อ่านเดือนไม่ได้ {bad:.1f}%{note}')
        dfs.append(df)

    dengue = pd.concat(dfs, ignore_index=True)
    dengue['province'] = dengue['province'].astype(str).str.strip()
    dengue['district'] = dengue['district'].astype(str).str.strip()
    return dengue


def load_population(valid_provs, district_agg):
    """
    อ่านไฟล์ประชากรระดับ ตำบล/หมู่บ้าน แล้วรวมเป็นรายอำเภอด้วยรหัสตำบล (ขึ้นต้น = จังหวัด2 หลัก + อำเภอ2 หลัก)
    วิธีนี้รวมประชากรทั้งในและนอกเขตเทศบาลของอำเภอเดียวกันได้ครบ
    (ไฟล์ระดับ 'สำนักทะเบียน' ใช้ไม่ได้ เพราะแถวเทศบาลไม่บอกว่าอยู่อำเภอไหน)
    """
    raw = pd.read_html(POP_FILE, header=0, encoding='utf-8')[0]
    if raw.shape[1] < 13:
        raise SystemExit(f'{POP_FILE} มี {raw.shape[1]} คอลัมน์ (คาดว่า 13 คอลัมน์ตามรูปแบบของกรมการปกครอง)')
    raw = raw.iloc[:, :13].copy()
    raw.columns = ['ym', 'pcode', 'province', 'ocode', 'office', 'tcode', 'tname',
                   'vcode', 'vname', 'male', 'female', 'total_people', 'total_houses']
    for col in ['pcode', 'ocode', 'tcode', 'vcode', 'total_people', 'total_houses']:
        raw[col] = to_num(raw[col])
    raw['province'] = raw['province'].astype(str).str.replace('จังหวัด', '', regex=False).str.strip()
    raw['office'] = raw['office'].astype(str).str.strip()
    raw = raw[raw['province'].isin(valid_provs)].copy()

    # ยอดรวมจังหวัด (ไว้ตรวจความถูกต้อง) = แถวที่รหัสสำนักทะเบียน/ตำบล/หมู่บ้านเป็น 0 ทั้งหมด
    prov_total = (raw[(raw['ocode'] == 0) & (raw['tcode'] == 0) & (raw['vcode'] == 0)]
                  .groupby('province')['total_people'].sum())

    detail = raw[raw['tcode'] > 0].copy()
    if detail.empty:
        raise SystemExit(
            f'\n❌ {POP_FILE} เป็นข้อมูลระดับ "สำนักทะเบียน" (คอลัมน์รหัสตำบล/หมู่บ้านเป็น 0 ทุกแถว)\n'
            '   ในไฟล์ระดับนี้ แถวเทศบาลไม่ระบุว่าอยู่อำเภอไหน ถ้าใช้จะนับประชากรอำเภอขาดไปมาก\n'
            '   (เช่น นนทบุรีใช้ได้เพียง 40% ของประชากรจริง) อัตราป่วยจึงสูงเกินจริง\n'
            '   -> กรุณาดาวน์โหลดข้อมูลชุดเดียวกัน (ปี 2566) ใหม่ โดยเลือกระดับ "ตำบล" หรือ "หมู่บ้าน"\n'
            f'      ให้คอลัมน์ "รหัสตำบล" มีค่า แล้วบันทึกเป็น {POP_FILE}')

    if (detail['vcode'] > 0).any():           # ระดับหมู่บ้าน
        detail = detail[detail['vcode'] > 0]
        level = 'หมู่บ้าน'
    else:                                      # ระดับตำบล
        level = 'ตำบล'
    # รหัสตำบลในไฟล์จริงมี 8 หลัก (จว.2 + อำเภอ2 + ตำบล2 + ท้าย2) เช่น 13010100 -> อำเภอ 1301
    # รองรับรหัส 6 หลักด้วย (130101 -> 1301)
    detail['district_code'] = np.where(detail['tcode'] >= 10_000_000,
                                       detail['tcode'] // 10000, detail['tcode'] // 100).astype(int)
    print(f'   ไฟล์ประชากรระดับ{level}: {len(detail):,} แถว')

    # ตรวจ: ผลรวมของแถวรายละเอียด ต้องเท่ากับยอดจังหวัด
    chk = detail.groupby('province')['total_people'].sum().to_frame('sum_detail').join(prov_total.rename('prov_total'))
    chk['diff_%'] = (chk['sum_detail'] - chk['prov_total']) / chk['prov_total'] * 100
    print('   ตรวจยอดรวมรายจังหวัด (รวมจากตำบล/หมู่บ้าน เทียบกับแถวยอดจังหวัด):')
    print(chk.round(2).to_string())
    if (chk['diff_%'].abs() > 2).any():
        raise SystemExit('\n❌ ยอดรวมจากตำบล/หมู่บ้านไม่ตรงกับยอดจังหวัดเกิน 2% (นับซ้ำหรือขาด) ตรวจรูปแบบไฟล์ก่อนใช้งาน')

    # ชื่ออำเภอ: จากแถวที่สำนักทะเบียนเป็น "อำเภอX" และรหัสสำนักทะเบียน = รหัสอำเภอ
    names = {}
    for _, r in raw[(raw['ocode'] > 0) & raw['office'].str.startswith('อำเภอ')].iterrows():
        names[(r['province'], int(r['ocode']))] = norm_name(r['office'])
    for code, nm in DISTRICT_CODE_NAMES.items():
        for prov in detail.loc[detail['district_code'] == code, 'province'].unique():
            names[(prov, code)] = norm_name(nm)

    agg = (detail.groupby(['province', 'district_code'], as_index=False)
                 .agg(total_people=('total_people', 'sum'), total_houses=('total_houses', 'sum')))
    agg['district_key'] = [names.get((p, c)) for p, c in zip(agg['province'], agg['district_code'])]

    # รหัสที่ยังไม่มีชื่อ: จับคู่โดยตัดออก ถ้าในจังหวัดนั้นเหลือรหัสเดียวและชื่อจากข้อมูลผู้ป่วยเหลือชื่อเดียว
    for prov, g in agg[agg['district_key'].isna()].groupby('province'):
        used = set(agg.loc[(agg['province'] == prov) & agg['district_key'].notna(), 'district_key'])
        cands = [k for k in district_agg.loc[district_agg['province'] == prov, 'district_key'] if k not in used]
        if len(g) == 1 and len(cands) == 1:
            agg.loc[g.index, 'district_key'] = cands[0]
            print(f'   ตั้งชื่อรหัส {int(g["district_code"].iloc[0])} ({prov}) = {cands[0]} (จากการตัดออก)')
        else:
            print(f'   ⚠ {prov}: รหัสอำเภอที่ไม่มีชื่อ {g["district_code"].astype(int).tolist()} '
                  f'ชื่อที่ยังไม่ถูกจับคู่ {cands} -> เพิ่มใน DISTRICT_CODE_NAMES ด้านบน')

    return agg.dropna(subset=['district_key'])[['province', 'district_key', 'total_people', 'total_houses']], prov_total


# =====================================================================
# Main
# =====================================================================
def main():
    print('1. Loading Dengue data (2561-2566)...')
    dengue = load_dengue()
    n_years = dengue['year_be'].nunique()

    if TARGET_PROVINCES:
        dengue = dengue[dengue['province'].isin(TARGET_PROVINCES)]

    print('\n2. Feature Engineering...')
    dengue['agey'] = pd.to_numeric(dengue['agey'], errors='coerce')

    # ค่า NaN = "ไม่ทราบ" ไม่ถูกนับเป็น 0 แล้ว (mean ของ pandas จะข้ามให้เอง)
    dengue['is_rainy'] = np.where(dengue['month'].isna(), np.nan,
                                  dengue['month'].isin(RAINY_MONTHS).astype(float))
    dengue['is_metro'] = np.where(dengue['metropol'].isna(), np.nan,
                                  (dengue['metropol'].astype(str).str.strip() == 'ในเขตเทศบาล').astype(float))
    dengue['is_youth'] = np.where(dengue['agey'].isna(), np.nan,
                                  (dengue['agey'] <= YOUTH_MAX_AGE).astype(float))

    print('\n   สัดส่วนเคสช่วงหน้าฝน แยกตามจังหวัด x ปี (แถว/คอลัมน์ที่ต่างจากที่อื่นมาก = วันที่อาจอ่านผิด):')
    print(dengue.pivot_table(index='province', columns='year_be', values='is_rainy',
                             aggfunc='mean').round(2).to_string())

    # รวมอำเภอที่สะกดต่างกัน (เช่น 'เมืองพระนครศรีอยุธยา' กับ 'พระนครศรีอยุธยา') ให้เป็นอำเภอเดียว "ก่อน" นับเคส
    # (ถ้าไม่รวม แต่ละชื่อจะได้ประชากรทั้งอำเภอซ้ำกัน และเคสถูกแบ่งครึ่ง อัตราป่วยจึงต่ำเกินจริง)
    dengue['district_key'] = [norm_name(DISTRICT_ALIASES.get((p, d), d))
                              for p, d in zip(dengue['province'], dengue['district'])]
    variants = dengue.groupby(['province', 'district_key'])['district'].unique()
    for (prov, _), names in variants.items():
        if len(names) > 1:
            print(f'   รวมชื่อที่สะกดต่างกัน {prov}: {list(names)}')

    def pick_display(names):
        """ชื่อที่ใช้แสดง: ถ้ามีรูป 'เมืองX' ใช้รูปนั้น ไม่งั้นใช้ชื่อที่พบบ่อยสุด (ไม่ใช้ .str เพื่อกันปัญหาข้ามเวอร์ชัน pandas)"""
        names = [str(n) for n in names if pd.notna(n)]
        if not names:
            return ''
        muang = [n for n in names if n.startswith('เมือง')]
        return Counter(muang if muang else names).most_common(1)[0][0]

    display = pd.DataFrame(
        [(prov, key, pick_display(g))
         for (prov, key), g in dengue.groupby(['province', 'district_key'])['district']],
        columns=['province', 'district_key', 'district'])

    district_agg = (dengue.groupby(['province', 'district_key'])
                    .agg(total_cases=('year_be', 'size'),     # นับทุกเคส (เดิมนับเฉพาะเคสที่มีอายุ)
                         is_metro=('is_metro', 'mean'),
                         is_rainy=('is_rainy', 'mean'),
                         is_youth=('is_youth', 'mean'))
                    .reset_index())
    district_agg = district_agg.merge(display, on=['province', 'district_key'])

    # ปีที่ระบาด = ปีที่มีเคส >= ENDEMIC_CASES
    yearly = dengue.groupby(['province', 'district_key', 'year_be']).size().reset_index(name='cases')
    endemic = (yearly[yearly['cases'] >= ENDEMIC_CASES]
               .groupby(['province', 'district_key']).size().reset_index(name='endemic_years'))
    district_agg = district_agg.merge(endemic, on=['province', 'district_key'], how='left')
    district_agg['endemic_years'] = district_agg['endemic_years'].fillna(0).astype(int)

    # ตัดชื่ออำเภอขยะ (มีตัวเลข / ว่าง)
    junk = (district_agg['district'].str.contains(r'\d', regex=True)
            | district_agg['district'].isin(['', 'nan', 'None']))
    if junk.any():
        print(f'   ตัดชื่ออำเภอที่ผิดปกติ {junk.sum()} แถว ({district_agg.loc[junk, "total_cases"].sum():,} เคส): '
              f'{district_agg.loc[junk, "district"].tolist()[:10]}')
    district_agg = district_agg[~junk].copy()

    print(f'\n3. Loading Population data ({POP_FILE})...')
    try:
        pop_agg, prov_total = load_population(district_agg['province'].unique(), district_agg)
    except Exception as e:
        raise SystemExit(f'อ่าน {POP_FILE} ไม่ได้: {e}')

    print('4. Merging (ไม่เติม median แทนค่าที่หายอีกต่อไป)...')
    merged = district_agg.merge(pop_agg, on=['province', 'district_key'], how='left')

    unmatched = merged[merged['total_people'].isna() | (merged['total_people'] <= 0)]
    if len(unmatched):
        unmatched[['province', 'district', 'total_cases']].to_csv(
            UNMATCHED_FILE, index=False, encoding='utf-8-sig')
        print(f'   ⚠ merge ประชากรไม่ติด {len(unmatched)} อำเภอ -> บันทึกที่ {UNMATCHED_FILE}')
        print(unmatched[['province', 'district', 'total_cases']].head(15).to_string(index=False))
        print('   คำแนะนำ (ชื่อในไฟล์ประชากรที่ยังไม่ถูกจับคู่ของจังหวัดนั้น):')
        for _, r in unmatched.iterrows():
            prov_pop = pop_agg[pop_agg['province'] == r['province']]
            if prov_pop.empty:
                print(f"   - {r['province']}/{r['district']}: ไม่พบจังหวัดนี้ในไฟล์ประชากร (เช็กการสะกดชื่อจังหวัด)")
                continue
            used = set(district_agg.loc[district_agg['province'] == r['province'], 'district_key'])
            cands = [k for k in prov_pop['district_key'] if k not in used]
            close = difflib.get_close_matches(r['district_key'], cands, n=3, cutoff=0.4)
            print(f"   - {r['province']}/{r['district']}: {close or cands[:10] or 'ไม่มีชื่อเหลือในไฟล์ประชากร'}")
        print('     -> ถ้าเจอชื่อที่ใช่ ให้เพิ่มใน DISTRICT_ALIASES ด้านบนแล้วรันใหม่')
    final_df = merged.drop(unmatched.index).drop(columns='district_key').copy()

    used = final_df.groupby('province')['total_people'].sum()
    pop_check = pd.concat([used.rename('ประชากรที่ใช้'), prov_total.rename('ยอดจังหวัด')], axis=1)
    pop_check['ใช้/ยอดจังหวัด_%'] = pop_check['ประชากรที่ใช้'] / pop_check['ยอดจังหวัด'] * 100
    print('\n   ประชากรที่ใช้ เทียบกับยอดจังหวัด (ต้องไม่เกิน 100% ถ้าเกินแปลว่านับอำเภอซ้ำ):')
    print(pop_check.round(1).to_string())
    over = pop_check[pop_check['ใช้/ยอดจังหวัด_%'] > 100.05]
    if len(over):
        raise SystemExit(f'\n❌ ประชากรที่ใช้เกินยอดจังหวัด {over.index.tolist()} -> มีอำเภอถูกนับซ้ำ '
                         '(ชื่ออำเภอสองแบบที่ยังไม่ถูกรวม) ตรวจรายชื่ออำเภอของจังหวัดนั้น')

    # อัตราป่วยสะสม (คอลัมน์เดิม ให้ api.py ใช้ต่อได้) + อัตราป่วยเฉลี่ยต่อปี
    final_df['incidence_rate'] = final_df['total_cases'] / final_df['total_people'] * 100000
    final_df['incidence_rate_annual'] = final_df['incidence_rate'] / n_years
    final_df['house_density'] = final_df['total_people'] / final_df['total_houses'].replace(0, np.nan)
    final_df['log_incidence'] = np.log1p(final_df['incidence_rate'])

    print(f'\n5. Running K-Means Clustering (k={K})...')
    eligible = final_df['total_cases'] >= MIN_CASES
    print(f'   เข้าโมเดล {eligible.sum()} อำเภอ | เคสน้อยกว่า {MIN_CASES} (ไม่เข้าโมเดล) {(~eligible).sum()} อำเภอ')

    train = final_df[eligible].copy()
    if len(train) <= K:
        raise SystemExit(f'อำเภอที่เข้าโมเดลมี {len(train)} แห่ง น้อยเกินไปสำหรับ k={K} ลองลด K หรือ MIN_CASES')

    X = train[FEATURES]
    X = X.fillna(X.median())                       # NaN -> median ของกลุ่ม (ไม่ใช่ 0)
    X_scaled = StandardScaler().fit_transform(X)

    print('   Silhouette score (ยิ่งสูงยิ่งแยกกลุ่มชัด):')
    best_k, best_s = None, -1
    for k in range(2, min(9, len(train))):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_scaled)
        s = silhouette_score(X_scaled, labels)
        if s > best_s:
            best_k, best_s = k, s
        mark = '  <-- k ที่ใช้' if k == K else ''
        print(f'     k={k}: {s:.3f}{mark}')
    print(f'   k ที่ silhouette สูงสุด = {best_k} ({best_s:.3f})')

    km = KMeans(n_clusters=K, random_state=42, n_init=20)
    train['cluster'] = km.fit_predict(X_scaled)

    # เรียง id ตามอัตราป่วย: 0 = เสี่ยงต่ำสุด ... K-1 = เสี่ยงสูงสุด (เลขคงที่ทุกครั้งที่รัน)
    order = train.groupby('cluster')['incidence_rate'].mean().sort_values().index
    train['cluster'] = train['cluster'].map({old: new for new, old in enumerate(order)})

    final_df['cluster'] = -1                       # -1 = ข้อมูลไม่เพียงพอ
    final_df['status'] = 'insufficient_data'
    final_df.loc[train.index, 'cluster'] = train['cluster']
    final_df.loc[train.index, 'status'] = 'clustered'

    # กัน NaN ในไฟล์ที่ api.py จะอ่าน (NaN ทำให้ JSON error)
    for col in ['is_metro', 'is_rainy', 'is_youth', 'house_density']:
        final_df[col] = final_df[col].fillna(0)

    out_cols = ['province', 'district', 'cluster', 'status', 'total_cases', 'total_people',
                'total_houses', 'house_density', 'incidence_rate', 'incidence_rate_annual',
                'is_metro', 'is_rainy', 'is_youth', 'endemic_years']
    final_df[out_cols].to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ Clustering complete! Results saved to '{OUTPUT_FILE}'")

    print('\n--- Cluster Summary (ค่าเฉลี่ยของแต่ละกลุ่ม) ---')
    cols = ['incidence_rate', 'incidence_rate_annual', 'is_metro', 'is_rainy', 'is_youth', 'endemic_years']
    summary = final_df[final_df['cluster'] >= 0].groupby('cluster')[cols].mean()
    summary['count'] = final_df[final_df['cluster'] >= 0].groupby('cluster').size()
    print(summary.round(2))

    small = summary[summary['count'] < 3]
    if len(small):
        print(f'\n⚠ cluster {small.index.tolist()} มีสมาชิกน้อยกว่า 3 อำเภอ ลองลด K')

    print('\n--- สมาชิกในแต่ละ cluster ---')
    for cid, g in final_df.groupby('cluster'):
        label = 'ข้อมูลไม่เพียงพอ' if cid == -1 else f'cluster {cid}'
        names = ', '.join((g['province'] + '/' + g['district']).head(12))
        more = f' ... (+{len(g) - 12})' if len(g) > 12 else ''
        print(f'[{label}] {len(g)} อำเภอ: {names}{more}')


if __name__ == '__main__':
    main()