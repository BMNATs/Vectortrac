import math
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DengueRisk AI API", version="4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_FILE = 'dengue_clustering_result.csv'
SEASONAL_FILE = 'dengue_seasonal_profile.csv'   # สร้างโดย seasonal_forecast.py
N_LEVELS = 5          # จำนวนระดับความเสี่ยง (แบ่งตามสัดส่วนของอำเภอที่มีข้อมูลเพียงพอ)


def custom_round(val: float) -> int:
    if math.isnan(val) or math.isinf(val):
        return 0
    return math.floor(val + 0.5)


# ถ้าต้องการจำกัดอำเภอของจังหวัดใด ให้ใส่ที่นี่ (จังหวัดที่ไม่อยู่ในรายการนี้จะแสดงทุกอำเภอ)
VALID_DISTRICTS = {
    "นนทบุรี": ["เมืองนนทบุรี", "บางกรวย", "บางใหญ่", "บางบัวทอง", "ไทรน้อย", "ปากเกร็ด"],
    "ปทุมธานี": ["เมืองปทุมธานี", "คลองหลวง", "ธัญบุรี", "หนองเสือ", "ลาดหลุมแก้ว", "ลำลูกกา", "สามโคก"]
}

# ======================================================================
# ส่วนที่ 1: ระดับความเสี่ยง (คิดจากอัตราป่วยเฉลี่ยต่อปีของอำเภอเอง จึงตรงกับลำดับในรายชื่อเสมอ)
# แบ่ง 5 ระดับเท่าๆ กันตามลำดับของทุกอำเภอที่มีข้อมูลเพียงพอ (เป็นระดับสัมพัทธ์ ไม่ใช่เกณฑ์ทางการ)
# ======================================================================
RISK_LEVELS = {
    4: {"label": "เสี่ยงสูงมาก", "color": "#dc2626"},
    3: {"label": "เสี่ยงสูง", "color": "#f97316"},
    2: {"label": "เสี่ยงปานกลาง", "color": "#eab308"},
    1: {"label": "เสี่ยงต่ำถึงปานกลาง", "color": "#84cc16"},
    0: {"label": "เสี่ยงต่ำ", "color": "#22c55e"},
    -1: {"label": "ข้อมูลไม่เพียงพอ", "color": "#94a3b8"},
}

RISK_TEXT = {
    4: {
        "summary": "เสี่ยงสูงมาก เมื่อเทียบกับอำเภออื่นในพื้นที่ศึกษา",
        "citizen": "จำนวนผู้ป่วยต่อประชากรอยู่ในกลุ่มสูงที่สุด ควรกำจัดแหล่งน้ำขังรอบบ้านทุกสัปดาห์ ทายากันยุง และหากมีไข้สูงลอยเกิน 2 วันควรรีบพบแพทย์",
        "official": "มาตรการเร่งด่วน: ตั้งทีมสอบสวนโรคเชิงรุก สำรวจดัชนีลูกน้ำยุงลาย (HI/CI) ติดตามผู้ป่วยรายสัปดาห์ และพิจารณาประกาศเตือนภัยในพื้นที่เมื่อเคสเพิ่มผิดปกติ",
    },
    3: {
        "summary": "เสี่ยงสูง เมื่อเทียบกับอำเภออื่นในพื้นที่ศึกษา",
        "citizen": "จำนวนผู้ป่วยต่อประชากรอยู่ในระดับสูง ควรกำจัดน้ำขังรอบบ้านสม่ำเสมอ ป้องกันยุงกัด และสังเกตอาการไข้สูงฉับพลัน",
        "official": "มาตรการเชิงรุก: สำรวจลูกน้ำยุงลายในชุมชนเสี่ยง ประสาน อสม. กำจัดแหล่งเพาะพันธุ์ และติดตามผู้ป่วยรายสัปดาห์",
    },
    2: {
        "summary": "เสี่ยงปานกลาง เมื่อเทียบกับอำเภออื่นในพื้นที่ศึกษา",
        "citizen": "จำนวนผู้ป่วยต่อประชากรอยู่ในระดับปานกลาง ควรกำจัดน้ำขังรอบบ้านเป็นประจำและป้องกันยุงกัด",
        "official": "เฝ้าระวังต่อเนื่อง: สนับสนุนการสำรวจลูกน้ำโดย อสม. และติดตามแนวโน้มผู้ป่วยรายเดือน",
    },
    1: {
        "summary": "เสี่ยงต่ำถึงปานกลาง เมื่อเทียบกับอำเภออื่นในพื้นที่ศึกษา",
        "citizen": "จำนวนผู้ป่วยต่อประชากรค่อนข้างต่ำ แต่ยังควรกำจัดน้ำขังรอบบ้านและป้องกันยุงกัดตามปกติ",
        "official": "เฝ้าระวังตามปกติ: ติดตามเคสรายเดือน และเตรียมวัสดุสำรองไว้ใช้เมื่อเคสเพิ่มผิดปกติ",
    },
    0: {
        "summary": "เสี่ยงต่ำ เมื่อเทียบกับอำเภออื่นในพื้นที่ศึกษา",
        "citizen": "จำนวนผู้ป่วยต่อประชากรต่ำเมื่อเทียบกับอำเภออื่น แต่ยังควรกำจัดน้ำขังรอบบ้านและป้องกันยุงกัดตามปกติ",
        "official": "รักษาการเฝ้าระวังตามปกติ และตรวจสอบคุณภาพการรายงานข้อมูลของพื้นที่",
    },
    -1: {
        "summary": "ข้อมูลในพื้นที่ยังไม่เพียงพอต่อการประเมินระดับความเสี่ยง (ผู้ป่วยรวม 6 ปีน้อยกว่า 20 ราย)",
        "citizen": "ขณะนี้มีข้อมูลในพื้นที่น้อย ยังไม่สามารถประเมินระดับเสี่ยงได้ ควรป้องกันยุงกัดและกำจัดน้ำขังรอบบ้านตามคำแนะนำทั่วไป",
        "official": "ตรวจสอบข้อมูล: ประสานสถานพยาบาลในพื้นที่เพื่อทบทวนการรายงานผู้ป่วยให้ครบถ้วน และติดตามต่อเนื่องจนมีข้อมูลเพียงพอ",
    },
}

# ======================================================================
# ส่วนที่ 2: รูปแบบการระบาด (ผลจัดกลุ่ม K-Means แยกต่างหาก ไม่ได้บอกว่าป่วยมากหรือน้อย)
# เลขกลุ่มเรียงตามอัตราป่วยเฉลี่ยของกลุ่มโดย dengue_clustering.py (0 ต่ำสุด ... 5 สูงสุด)
# ข้อความด้านล่างเขียนจากตาราง Cluster Summary ของการรัน k=6 รอบปัจจุบัน
# ถ้าเปลี่ยน K หรือข้อมูล ต้องทบทวนข้อความนี้ใหม่
#   กลุ่ม | ในเขตเทศบาล | หน้าฝน | อายุ<=24 | ปีระบาดเฉลี่ย
#     5   |     12%     |  72%   |   76%    |  3.3
#     4   |     56%     |  67%   |   73%    |  4.3
#     3   |     38%     |  66%   |   68%    |  1.9
#     2   |     57%     |  56%   |   61%    |  4.8
#     1   |     38%     |  51%   |   54%    |  2.1
#     0   |     21%     |  49%   |   78%    |  0.8
# ======================================================================
PATTERNS = {
    5: {
        "title": "ชนบท ชุกตามฤดูฝน ผู้ป่วยส่วนใหญ่เป็นเด็กและเยาวชน",
        "detail": "ค่าเฉลี่ยของกลุ่ม: ผู้ป่วยอยู่ในเขตเทศบาลเพียง 12% ป่วยในหน้าฝน 72% และอายุไม่เกิน 24 ปี 76% ระบาดเฉลี่ย 3.3 จาก 6 ปี",
        "citizen": "ผู้ป่วยส่วนใหญ่เป็นเด็กและเยาวชนและพบมากในหน้าฝน ควรปิดฝาโอ่งหรือถังน้ำ คว่ำภาชนะที่มีน้ำขัง และทายากันยุงให้เด็กก่อนไปโรงเรียน",
        "official": "รูปแบบชนบทชุกตามฤดูฝน: ควบคุมแหล่งเพาะพันธุ์ในโรงเรียนและศูนย์พัฒนาเด็กเล็ก แจกทรายอะเบทสำหรับภาชนะกักเก็บน้ำ และให้ อสม. ลงพื้นที่นอกเขตเทศบาลก่อนต้นฤดูฝน (พ.ค.-มิ.ย.)",
    },
    4: {
        "title": "เมือง ระบาดซ้ำหลายปี ชุกตามฤดูฝน",
        "detail": "ค่าเฉลี่ยของกลุ่ม: ผู้ป่วยอยู่ในเขตเทศบาล 56% ป่วยในหน้าฝน 67% และอายุไม่เกิน 24 ปี 73% ระบาดเฉลี่ย 4.3 จาก 6 ปี",
        "citizen": "พื้นที่นี้พบการระบาดซ้ำหลายปีและเพิ่มขึ้นในหน้าฝน ควรกำจัดน้ำขังรอบบ้านและส่วนกลางของหมู่บ้านหรืออาคารเป็นประจำ",
        "official": "รูปแบบเขตเมืองระบาดซ้ำ: สำรวจหมู่บ้านจัดสรร คอนโดมิเนียม ตลาด และสถานที่ก่อสร้าง ลอกท่อระบายน้ำร่วมกับเทศบาล และเร่งรณรงค์ก่อนต้นฤดูฝน",
    },
    3: {
        "title": "ชุกตามฤดูฝน ระบาดเป็นบางปี",
        "detail": "ค่าเฉลี่ยของกลุ่ม: ผู้ป่วยอยู่ในเขตเทศบาล 38% ป่วยในหน้าฝน 66% และอายุไม่เกิน 24 ปี 68% ระบาดเฉลี่ย 1.9 จาก 6 ปี",
        "citizen": "พื้นที่นี้มักพบการระบาดเป็นบางปี โดยเฉพาะหน้าฝน แม้ปีก่อนจะเงียบก็ไม่ควรประมาท",
        "official": "รูปแบบระบาดเป็นบางปี: เฝ้าระวังเคสรายสัปดาห์ในช่วงฝนเพื่อจับสัญญาณเตือน รณรงค์ก่อนต้นฤดูฝน และเตรียมทีมเคลื่อนที่เร็ว",
    },
    2: {
        "title": "เมือง ระบาดซ้ำเกือบทุกปี ป่วยได้ตลอดปี",
        "detail": "ค่าเฉลี่ยของกลุ่ม: ผู้ป่วยอยู่ในเขตเทศบาล 57% ป่วยในหน้าฝน 56% และอายุไม่เกิน 24 ปี 61% ระบาดเฉลี่ย 4.8 จาก 6 ปี",
        "citizen": "พื้นที่นี้พบผู้ป่วยเกือบทุกปีและพบได้แม้นอกหน้าฝน รวมถึงคนวัยทำงาน ควรกำจัดน้ำขังทั้งที่บ้านและที่ทำงานตลอดปี",
        "official": "รูปแบบเขตเมืองป่วยตลอดปี: ตรวจสถานประกอบการ อาคารสำนักงาน และสถานที่ก่อสร้าง ลอกท่อร่วมกับเทศบาล และให้ อสม. สำรวจลูกน้ำต่อเนื่องไม่เฉพาะหน้าฝน",
    },
    1: {
        "title": "กระจายตลอดปี ผู้ใหญ่ป่วยสัดส่วนสูง",
        "detail": "ค่าเฉลี่ยของกลุ่ม: ผู้ป่วยอยู่ในเขตเทศบาล 38% ป่วยในหน้าฝน 51% และอายุไม่เกิน 24 ปี 54% ระบาดเฉลี่ย 2.1 จาก 6 ปี",
        "citizen": "ผู้ป่วยกระจายตลอดปีและมีผู้ใหญ่ป่วยสัดส่วนสูงกว่าที่อื่น ควรป้องกันยุงกัดทั้งเด็กและผู้ใหญ่",
        "official": "รูปแบบกระจายตลอดปี: ติดตามเคสรายสัปดาห์ตลอดปี ไม่เน้นเฉพาะหน้าฝน และสนับสนุน อสม. สำรวจลูกน้ำต่อเนื่อง",
    },
    0: {
        "title": "ชนบท ระบาดไม่ต่อเนื่อง",
        "detail": "ค่าเฉลี่ยของกลุ่ม: ผู้ป่วยอยู่ในเขตเทศบาล 21% ป่วยในหน้าฝน 49% และอายุไม่เกิน 24 ปี 78% ระบาดเฉลี่ยเพียง 0.8 จาก 6 ปี",
        "citizen": "ผู้ป่วยส่วนใหญ่เป็นเด็กและเยาวชน และพื้นที่นี้ไม่ได้ระบาดต่อเนื่องทุกปี แต่อาจเกิดการระบาดเป็นบางปีได้ ควรให้เด็กป้องกันยุงกัดและกำจัดน้ำขังรอบบ้าน",
        "official": "รูปแบบระบาดไม่ต่อเนื่อง: ควบคุมยุงลายในโรงเรียนและศูนย์พัฒนาเด็กเล็ก และเตรียมรับปีที่เคสเพิ่มขึ้นเป็นระยะ",
    },
    -1: {
        "title": "ข้อมูลไม่เพียงพอสำหรับการจัดกลุ่มรูปแบบ",
        "detail": "ผู้ป่วยรวม 6 ปีน้อยกว่า 20 ราย จึงไม่นำเข้าโมเดลจัดกลุ่ม",
        "citizen": "",
        "official": "",
    },
}
FALLBACK_PATTERN = {"title": "ไม่ทราบรูปแบบ", "detail": "", "citizen": "", "official": ""}


def load_clean_df():
    df = pd.read_csv(CSV_FILE)
    df['district_clean'] = df['district'].astype(str).str.strip()
    df = df[~df['district_clean'].str.contains(r'\d', regex=True)].copy()
    if 'incidence_rate_annual' not in df.columns:
        df['incidence_rate_annual'] = df['incidence_rate'] / 6
    return df


def add_risk_levels(df: pd.DataFrame):
    """คำนวณระดับความเสี่ยงจากอัตราป่วยเฉลี่ยต่อปี เทียบทุกอำเภอที่มีข้อมูลเพียงพอ (ก่อนกรองจังหวัด)"""
    df = df.copy()
    df['risk_level'] = -1
    df['rank_overall'] = 0
    ok = df['cluster'] >= 0
    scale, total = [], int(ok.sum())
    if total >= N_LEVELS:
        s = df.loc[ok, 'incidence_rate_annual']
        # ใช้ลำดับ (rank) ก่อนแบ่ง เพื่อให้แต่ละระดับมีจำนวนเท่าๆ กัน แม้ค่าซ้ำกัน
        df.loc[ok, 'risk_level'] = pd.qcut(s.rank(method='first'), N_LEVELS, labels=False).astype(int)
        df.loc[ok, 'rank_overall'] = s.rank(method='min', ascending=False).astype(int)
        for lvl in range(N_LEVELS - 1, -1, -1):
            part = df.loc[ok & (df['risk_level'] == lvl), 'incidence_rate_annual']
            scale.append({
                "level": lvl,
                "label": RISK_LEVELS[lvl]["label"],
                "color": RISK_LEVELS[lvl]["color"],
                "min": round(float(part.min()), 1),
                "max": round(float(part.max()), 1),
                "count": int(len(part)),
            })
    return df, scale, total


# ======================================================================
# ส่วนที่ 3: จำลองสถานการณ์ (what-if)
# ไม่ใช่การพยากรณ์ เป็นการตอบว่า "ถ้าเกิดปัจจัยเหล่านี้ อัตราป่วยและระดับความเสี่ยงจะเปลี่ยนอย่างไร"
# ตัวคูณทุกตัวมีค่า (ต่ำ, กลาง, สูง) เพื่อแสดงเป็นช่วงความไม่แน่นอน
# ======================================================================
RAIN_BASE_CM = 18.0   # ปริมาณฝนเฉลี่ยต่อเดือนในหน้าฝน (ซม.) ใช้แปลง "ฝน +X%" เป็นหน่วยซม. ของงานวิจัย
                      # ** ค่าประมาณของผู้พัฒนา ยังไม่ได้ตรวจกับข้อมูลกรมอุตุนิยมวิทยา ควรแก้ให้ตรงกับพื้นที่ **

# Phanitchat et al. 2019, BMC Infect Dis, จ.ขอนแก่น 2549-2559 (Bayesian Poisson, ข้อมูลรายเดือนรายตำบล)
RAIN_RR_PER_CM = (0.990, 1.004, 1.017)    # อัตราป่วยเพิ่มขึ้นต่อฝน 1 ซม./เดือน (ช่วงความเชื่อมั่น 95% คร่อม 1 = ไม่มีนัยสำคัญ)
TEMP_RR_PER_C = (1.009, 1.055, 1.115)     # อัตราป่วยเพิ่มขึ้นต่ออุณหภูมิสูงสุด 1 องศา/เดือน (มีนัยสำคัญ)
# "Impact of SARS-CoV-2 interventions on dengue transmission" PLoS NTD 2020: มาตรการเว้นระยะห่าง ผู้ป่วยเพิ่ม 4.32 ต่อแสนประชากรต่อเดือน (ทั้งประเทศ 2008 ราย/เดือน, 95% CI 1170-2846)
# ช่วงต่ำ/สูงคำนวณโดยสเกลจากช่วงความเชื่อมั่นระดับประเทศ (4.32 x 1170/2008 และ 4.32 x 2846/2008)
LOCKDOWN_PER_100K_MONTH = (2.5, 4.32, 6.1)

SIM_LIMITS = {"rain": (-50, 100), "temp": (-2, 4), "population": (-20, 50), "lockdown": (0, 12), "custom": (-50, 100)}

SIM_ASSUMPTIONS = [
    {
        "key": "rain",
        "title": "ฝนเปลี่ยนไป",
        "used": f"ฝน +1 ซม./เดือน ทำให้ผู้ป่วยเปลี่ยน 0.4% (ฝนปกติในหน้าฝนตั้งไว้ที่ราว {RAIN_BASE_CM:.0f} ซม./เดือน)",
        "range": "ช่วงจากงานวิจัย: -1.0% ถึง +1.7% ต่อ 1 ซม. (ผลไม่มีนัยสำคัญทางสถิติ)",
        "note": "ใช้กับสัดส่วนผู้ป่วยในเดือนหน้าฝนของแต่ละอำเภอเท่านั้น ค่าฝนปกติ 18 ซม. เป็นค่าประมาณที่ยังไม่ได้ตรวจกับข้อมูลกรมอุตุนิยมวิทยา และงานวิจัยไม่ได้ศึกษาผลแบบหน่วงเวลาหรือแบบไม่เป็นเส้นตรง",
        "source": "Phanitchat et al. 2019, BMC Infectious Diseases (จ.ขอนแก่น 2549-2559)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6708185",
    },
    {
        "key": "temp",
        "title": "อุณหภูมิเปลี่ยนไป",
        "used": "อุณหภูมิ +1 องศา ทำให้ผู้ป่วยเพิ่มขึ้น 5.5%",
        "range": "ช่วงจากงานวิจัย: +0.9% ถึง +11.5% ต่อ 1 องศา",
        "note": "เป็นปัจจัยภูมิอากาศเดียวที่งานวิจัยนี้พบผลชัดเจน ใช้กับผู้ป่วยทุกเดือน",
        "source": "Phanitchat et al. 2019, BMC Infectious Diseases (จ.ขอนแก่น 2549-2559)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6708185",
    },
    {
        "key": "lockdown",
        "title": "ล็อคดาวน์",
        "used": "ผู้ป่วยเพิ่มขึ้น 4.32 รายต่อประชากร 100,000 คน ต่อทุกเดือนที่ล็อคดาวน์",
        "range": "ช่วงที่ใช้: 2.5 ถึง 6.1 รายต่อ 100,000 คนต่อเดือน",
        "note": "ผลออกมาว่าล็อคดาวน์ทำให้ผู้ป่วยเพิ่มขึ้น ไม่ใช่ลดลง (คนอยู่บ้านมากขึ้น) ประเมินจากข้อมูลระดับประเทศถึงเดือนพฤษภาคม 2563 และผู้เขียนเตือนว่าการรายงานผู้ป่วยอาจตกหล่นช่วงล็อคดาวน์ ตัวเลขจริงอาจสูงกว่านี้ ค่านี้เป็นค่าเฉลี่ยทั้งประเทศ กรุงเทพฯ ได้ผลสูงกว่า",
        "source": "Impact of SARS-CoV-2 interventions on dengue transmission, PLoS Neglected Tropical Diseases 2020",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7595279",
    },
    {
        "key": "population",
        "title": "ประชากรเปลี่ยนไป",
        "used": "จำนวนผู้ป่วยเปลี่ยนตามสัดส่วนประชากร อัตราป่วยต่อ 100,000 คนไม่เปลี่ยน",
        "range": "ไม่มีช่วงความไม่แน่นอน",
        "note": "งานวิจัยภาพรวมพบว่าเมืองที่หนาแน่นมีความเสี่ยงสูงกว่า แต่ไม่ได้ให้ขนาดผลที่นำมาใช้คำนวณได้ จึงไม่รวมผลจากความแออัดเพิ่ม ผลจริงจากประชากรเพิ่มอาจสูงกว่าที่แสดง บ้านเรือนสมมติว่าเพิ่มตามสัดส่วนเดียวกัน",
        "source": "The role of urbanisation in the spread of Aedes mosquitoes and the diseases they transmit: a systematic review, PLoS NTD 2021 (ใช้เฉพาะทิศทางของผล)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8428665",
    },
    {
        "key": "custom",
        "title": "ปัจจัยอื่นที่ทีมประเมินเอง",
        "used": "ผู้ใช้กำหนดเปอร์เซ็นต์ที่ผู้ป่วยเปลี่ยนไปเอง",
        "range": "ไม่มีช่วงความไม่แน่นอน",
        "note": "ใช้กับพายุ น้ำท่วมขัง หรือโรคอื่นระบาด ที่ค้นแล้วไม่พบงานวิจัยที่ให้ตัวเลขนำมาใช้ได้ ผลที่ได้มาจากตัวเลขที่ผู้ใช้ตั้งเองทั้งหมด",
        "source": "ไม่มี",
        "url": "",
    },
]


def risk_cutpoints(df: pd.DataFrame):
    """เกณฑ์ตัดระดับความเสี่ยงจากสถานการณ์ปกติ (กึ่งกลางระหว่างขอบบนของระดับล่างกับขอบล่างของระดับบน) ล็อกไว้ใช้เทียบกับสถานการณ์จำลอง"""
    ok = df[df['risk_level'] >= 0]
    cuts = []
    for lvl in range(N_LEVELS - 1):
        lo = ok.loc[ok['risk_level'] == lvl, 'incidence_rate_annual'].max()
        hi = ok.loc[ok['risk_level'] == lvl + 1, 'incidence_rate_annual'].min()
        cuts.append(float((lo + hi) / 2))
    return cuts


def level_from_rate(rate: float, cuts) -> int:
    return int(sum(rate >= c for c in cuts))


def scenario_rate(base_rate, rainy_share, i, rain, temp, lockdown_months, custom):
    """อัตราป่วยต่อปีหลังจำลอง ตามชุดพารามิเตอร์ลำดับที่ i (0 = ต่ำ, 1 = กลาง, 2 = สูง)"""
    d_cm = rain / 100.0 * RAIN_BASE_CM
    rain_mult = 1.0 + rainy_share * (RAIN_RR_PER_CM[i] ** d_cm - 1.0)
    temp_mult = TEMP_RR_PER_C[i] ** temp
    rate = base_rate * rain_mult * temp_mult * (1.0 + custom / 100.0)
    rate += LOCKDOWN_PER_100K_MONTH[i] * lockdown_months
    return max(rate, 0.0)


@app.get("/api/simulate")
def simulate(
    province: Optional[str] = Query(None),
    rain: float = Query(0, ge=SIM_LIMITS["rain"][0], le=SIM_LIMITS["rain"][1]),
    temp: float = Query(0, ge=SIM_LIMITS["temp"][0], le=SIM_LIMITS["temp"][1]),
    population: float = Query(0, ge=SIM_LIMITS["population"][0], le=SIM_LIMITS["population"][1]),
    lockdown_months: float = Query(0, ge=SIM_LIMITS["lockdown"][0], le=SIM_LIMITS["lockdown"][1]),
    custom: float = Query(0, ge=SIM_LIMITS["custom"][0], le=SIM_LIMITS["custom"][1]),
):
    try:
        df, _, _ = add_risk_levels(load_clean_df())
        cuts = risk_cutpoints(df)

        if province and province.strip() != "":
            prov_target = province.strip()
            df = df[df['province'] == prov_target]
            if prov_target in VALID_DISTRICTS:
                df = df[df['district_clean'].isin(VALID_DISTRICTS[prov_target])]

        def side(rate, cases):
            lvl = level_from_rate(rate, cuts)
            return {"rate": round(rate, 1), "cases": round(cases, 1), "level": lvl,
                    "label": RISK_LEVELS[lvl]["label"], "color": RISK_LEVELS[lvl]["color"]}

        pop_mult = 1.0 + population / 100.0
        districts = []
        sum_before = sum_after = sum_low = sum_high = 0.0
        up_names, down_names = [], []

        for index, row in df.iterrows():
            name = str(row['district_clean'])
            people = float(row['total_people'])
            base_rate = float(row['incidence_rate_annual'])
            item = {"id": f"dist_{index}_{name}", "name": name, "province": str(row['province']),
                    "people": int(people),
                    # ใช้วาดฉากจำลองของอำเภอ (หน่วย %)
                    "metroRatio": round(float(row['is_metro']) * 100, 1),
                    "rainyRatio": round(float(row['is_rainy']) * 100, 1),
                    "youthRatio": round(float(row['is_youth']) * 100, 1),
                    "endemicYears": int(row['endemic_years'])}

            if int(row['risk_level']) < 0:
                item.update({"simulated": False, "change": "same"})
                districts.append(item)
                continue

            share = float(row['is_rainy']) if float(row['is_rainy']) > 0 else 0.5
            base_cases = base_rate * people / 100000.0
            rates = [scenario_rate(base_rate, share, i, rain, temp, lockdown_months, custom) for i in range(3)]
            r_mid = rates[1]
            r_low, r_high = min(rates[0], rates[2]), max(rates[0], rates[2])
            r_low, r_high = min(r_low, r_mid), max(r_high, r_mid)
            scale = people * pop_mult / 100000.0

            base_side = side(base_rate, base_cases)
            scen = side(r_mid, r_mid * scale)
            scen.update({
                "rateLow": round(r_low, 1), "rateHigh": round(r_high, 1),
                "casesLow": round(r_low * scale, 1), "casesHigh": round(r_high * scale, 1),
                "levelLow": level_from_rate(r_low, cuts), "levelHigh": level_from_rate(r_high, cuts),
                "people": int(round(people * pop_mult)),
            })
            change = "up" if scen["level"] > base_side["level"] else ("down" if scen["level"] < base_side["level"] else "same")
            if change == "up":
                up_names.append(name)
            elif change == "down":
                down_names.append(name)

            sum_before += base_cases
            sum_after += r_mid * scale
            sum_low += r_low * scale
            sum_high += r_high * scale
            item.update({"simulated": True, "baseline": base_side, "scenario": scen, "change": change})
            districts.append(item)

        scale_info = []
        for lvl in range(N_LEVELS - 1, -1, -1):
            scale_info.append({
                "level": lvl, "label": RISK_LEVELS[lvl]["label"], "color": RISK_LEVELS[lvl]["color"],
                "from": round(cuts[lvl - 1], 1) if lvl > 0 else None,
                "to": round(cuts[lvl], 1) if lvl < N_LEVELS - 1 else None,
            })

        return {
            "status": "success",
            "params": {"rain": rain, "temp": temp, "population": population,
                       "lockdownMonths": lockdown_months, "custom": custom},
            "summary": {
                "districts": sum(1 for d in districts if d["simulated"]),
                "up": len(up_names), "down": len(down_names),
                "upNames": up_names, "downNames": down_names,
                "casesBefore": round(sum_before, 1), "casesAfter": round(sum_after, 1),
                "casesLow": round(sum_low, 1), "casesHigh": round(sum_high, 1),
            },
            "cutpoints": scale_info,
            "assumptions": SIM_ASSUMPTIONS,
            "rainBaseCm": RAIN_BASE_CM,
            "districts": districts,
        }

    except FileNotFoundError:
        return {"status": "error", "message": f"ไม่พบไฟล์ {CSV_FILE}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/provinces")
def get_provinces():
    try:
        df = load_clean_df()
        provinces = sorted(df['province'].dropna().unique().tolist())
        return {"status": "success", "data": provinces}
    except FileNotFoundError:
        return {"status": "error", "message": f"ไม่พบไฟล์ {CSV_FILE}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/districts")
def get_districts_data(province: Optional[str] = Query(None)):
    try:
        df, risk_scale, total_ranked = add_risk_levels(load_clean_df())

        if province and province.strip() != "":
            prov_target = province.strip()
            df = df[df['province'] == prov_target]
            if prov_target in VALID_DISTRICTS:
                df = df[df['district_clean'].isin(VALID_DISTRICTS[prov_target])]

        result_data = []

        for index, row in df.iterrows():
            cluster_id = int(row['cluster'])
            risk = int(row['risk_level'])
            risk_info = RISK_LEVELS.get(risk, RISK_LEVELS[-1])
            risk_text = RISK_TEXT.get(risk, RISK_TEXT[-1])
            pattern = PATTERNS.get(cluster_id, FALLBACK_PATTERN)

            dist_name = str(row['district_clean'])
            prov_name = str(row['province'])

            total_people = int(row.get('total_people', 0))
            total_houses = int(row.get('total_houses', 0))
            total_cases = int(row.get('total_cases', 0))
            house_density = custom_round(float(row.get('house_density', 0)))

            # incidence_rate = อัตราป่วยสะสม 2561-2566 ต่อแสนประชากร
            incidence_rate = round(float(row['incidence_rate']), 1)
            incidence_annual = round(float(row['incidence_rate_annual']), 1)
            cases_per_1k = round(incidence_rate / 100, 2)

            radar_incidence = min((incidence_rate / 1500) * 100, 100)
            radar_metro = round(row['is_metro'] * 100, 1)
            radar_rainy = round(row['is_rainy'] * 100, 1)
            radar_youth = round(row['is_youth'] * 100, 1)
            radar_endemic = min((row['endemic_years'] / 6) * 100, 100)

            item = {
                "id": f"dist_{index}_{dist_name}",
                "province": prov_name,
                "name": dist_name,
                "status": str(row.get('status', 'clustered')),

                # ระดับความเสี่ยง (จากอัตราป่วย)
                "riskLevel": risk,
                "riskLabel": risk_info["label"],
                "color": risk_info["color"],
                "riskSummary": risk_text["summary"],
                "rankOverall": int(row['rank_overall']),

                # รูปแบบการระบาด (จากการจัดกลุ่ม)
                "cluster": cluster_id,
                "patternTitle": pattern["title"],
                "patternDetail": pattern["detail"],

                # คำแนะนำ = ตามระดับความเสี่ยง + ตามรูปแบบการระบาด
                "citizenAdvice": f'{risk_text["citizen"]} {pattern["citizen"]}'.strip(),
                "officialAction": f'{risk_text["official"]} {pattern["official"]}'.strip(),

                "totalCases": total_cases,
                "totalPeople": total_people,
                "totalHouses": total_houses,
                "houseDensity": house_density,
                "incidenceRate": incidence_rate,
                "incidenceAnnual": incidence_annual,
                "casesPer1k": cases_per_1k,
                "endemicYears": int(row['endemic_years']),
                "metroRatio": radar_metro,
                "rainyRatio": radar_rainy,
                "youthRatio": radar_youth,

                "radar": [
                    round(radar_incidence, 1),
                    radar_metro,
                    radar_rainy,
                    radar_youth,
                    round(radar_endemic, 1)
                ]
            }
            result_data.append(item)

        return {
            "status": "success",
            "total_districts": len(result_data),
            "totalRanked": total_ranked,
            "riskScale": risk_scale,
            "data": result_data
        }

    except FileNotFoundError:
        return {"status": "error", "message": f"ไม่พบไฟล์ {CSV_FILE}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======================================================================
# ปฏิทินเฝ้าระวังตามฤดูกาล (Seasonal Outlook)
# สรุปจากข้อมูลย้อนหลัง ไม่ใช่การพยากรณ์จำนวนผู้ป่วย
# ======================================================================
MONTH_TH = ['ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.']
WATCH_FACTOR = 1.2    # เดือนที่ผู้ป่วยสูงกว่าค่าเฉลี่ยรายเดือน (1/12 ของทั้งปี) อย่างน้อย 20% ถือเป็นช่วงเฝ้าระวัง


def _profile(df: pd.DataFrame, level: str, name: str):
    d = df[(df['level'] == level) & (df['name'].astype(str) == str(name))].sort_values('month')
    if len(d) != 12:
        return None
    return {
        "mean": [round(float(v) * 100, 1) for v in d['share_mean']],
        "min": [round(float(v) * 100, 1) for v in d['share_min']],
        "max": [round(float(v) * 100, 1) for v in d['share_max']],
        "nYears": int(d['n_years'].iloc[0]),
    }


def _month_ranges(months):
    """[9, 10, 11] -> 'ก.ย.–พ.ย.' (รวมช่วงที่ข้ามปี เช่น ธ.ค.–ก.พ.)"""
    ms = sorted(set(months))
    if not ms:
        return ""
    groups, cur = [], [ms[0]]
    for m in ms[1:]:
        if m == cur[-1] + 1:
            cur.append(m)
        else:
            groups.append(cur)
            cur = [m]
    groups.append(cur)
    if len(groups) > 1 and groups[0][0] == 1 and groups[-1][-1] == 12:
        groups[0] = groups.pop() + groups[0]
    out = []
    for g in groups:
        out.append(MONTH_TH[g[0] - 1] if len(g) == 1 else f"{MONTH_TH[g[0] - 1]}–{MONTH_TH[g[-1] - 1]}")
    return ", ".join(out)


@app.get("/api/seasonal")
def get_seasonal(province: str = Query(...), cluster: int = Query(-1)):
    try:
        df = pd.read_csv(SEASONAL_FILE)
    except FileNotFoundError:
        return {"status": "error", "message": f"ไม่พบไฟล์ {SEASONAL_FILE} (รัน seasonal_forecast.py ก่อน)"}
    try:
        prov = _profile(df, 'province', province.strip())
        clu = _profile(df, 'cluster', cluster) if cluster >= 0 else None
        basis_profile = clu or prov
        if basis_profile is None:
            return {"status": "error", "message": "ไม่มีข้อมูลฤดูกาลของพื้นที่นี้"}
        mean_share = 100 / 12
        watch = [i + 1 for i, v in enumerate(basis_profile["mean"]) if v >= mean_share * WATCH_FACTOR]
        peak = int(max(range(12), key=lambda i: basis_profile["mean"][i])) + 1
        return {
            "status": "success",
            "months": MONTH_TH,
            "cluster": clu,
            "province": prov,
            "basis": "cluster" if clu else "province",
            "avgShare": round(mean_share, 1),
            "watchThreshold": round(mean_share * WATCH_FACTOR, 1),
            "watchMonths": watch,
            "watchLabel": _month_ranges(watch),
            "peakMonth": peak,
            "peakLabel": MONTH_TH[peak - 1],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}