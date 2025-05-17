# -*- coding: utf-8 -*-
"""
Created on Sat May 17 17:29:06 2025

@author: ktrpt
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="3D Thumb-Index Viewer", layout="centered")
st.title("📐 3D Distance Between Thumb and Index (CSV from MediaPipe)")

uploaded_file = st.file_uploader("📄 Upload CSV of 21 hand landmarks (x/y/z)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if df.shape[0] != 21 or not all(col in df.columns for col in ['x', 'y', 'z']):
        st.error("CSV must have 21 rows and x/y/z columns.")
    else:
        # 親指と人差し指の先端インデックス
        thumb_tip = df.iloc[4].to_numpy()
        index_tip = df.iloc[8].to_numpy()

        # 座標反転処理（映像と一致させるため）
        df['x'] = 1.0 - df['x']
        df['y'] = 1.0 - df['y']
        df['z'] = -df['z']

        xs, ys, zs = df['x'], df['y'], df['z']
        distance = np.linalg.norm(index_tip - thumb_tip)

        # --- 3D プロット ---
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, zs, ys, c='blue', s=50)

        # 骨格接続線 (MediaPipeと同じ)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        for p1, p2 in connections:
            ax.plot([xs[p1], xs[p2]], [zs[p1], zs[p2]], [ys[p1], ys[p2]], 'gray')

        # 距離ベクトルの描画
        x0, y0, z0 = 1 - thumb_tip[0], 1 - thumb_tip[1], -thumb_tip[2]
        x1, y1, z1 = 1 - index_tip[0], 1 - index_tip[1], -index_tip[2]
        ax.quiver(x0, z0, y0, x1 - x0, z1 - z0, y1 - y0, color='red', linewidth=2)

        ax.set_xlabel("X (left-right)")
        ax.set_ylabel("Z (depth)")
        ax.set_zlabel("Y (up-down)")
        ax.view_init(elev=10, azim=70)
        st.pyplot(fig)

        st.markdown(f"### 📏 Distance between Thumb and Index Tip: `{distance:.3f}` units")
        st.markdown("### 🔢 Landmark Coordinates")
        st.dataframe(df)

        # ダウンロードボタン
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Processed CSV", csv, file_name="hand_landmarks_corrected.csv", mime="text/csv")
