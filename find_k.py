import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("1. กำลังโหลดข้อมูลจาก dengue_clustering_result.csv ...")
    try:
        df = pd.read_csv('dengue_clustering_result.csv')
    except FileNotFoundError:
        print("ไม่พบไฟล์! กรุณารันโค้ด dengue_clustering.py ก่อนครับ")
        return

    # เลือกตัวแปรเดิมที่เราใช้ทำ Clustering
    features = ['incidence_rate', 'is_metro', 'is_rainy', 'is_youth', 'endemic_years']
    X = df[features].fillna(0)

    # Standardize data เหมือนที่ทำในไฟล์หลัก
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # เตรียมเก็บค่าเพื่อวาดกราฟ
    inertia = []
    silhouette_avg = []
    K_range = range(2, 11) # ทดลองตั้งแต่ 2 ถึง 10 กลุ่ม

    print("2. กำลังคำนวณ Elbow Method และ Silhouette Score (K=2 ถึง 10)...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        
        # เก็บค่า Inertia (สำหรับ Elbow)
        inertia.append(kmeans.inertia_)
        
        # เก็บค่า Silhouette
        silhouette_avg.append(silhouette_score(X_scaled, kmeans.labels_))

    print("3. กำลังแสดงผลกราฟ...")
    # ตั้งค่ากราฟให้ออกมา 2 หน้าต่างคู่กัน
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # วาดกราฟ 1: Elbow Method
    ax[0].plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
    ax[0].set_title('Elbow Method (Find the "Bend")', fontsize=14)
    ax[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    ax[0].set_ylabel('Inertia (Sum of Squared Distances)', fontsize=12)
    ax[0].grid(True, linestyle='--', alpha=0.7)

    # วาดกราฟ 2: Silhouette Score
    ax[1].plot(K_range, silhouette_avg, 'gs-', linewidth=2, markersize=8)
    ax[1].set_title('Silhouette Score (Find the Peak)', fontsize=14)
    ax[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    ax[1].set_ylabel('Silhouette Score', fontsize=12)
    ax[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show() # หน้าต่างรูปกราฟจะเด้งขึ้นมา

if __name__ == '__main__':
    main()