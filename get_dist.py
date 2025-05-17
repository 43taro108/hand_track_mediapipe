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
import io

# Mediapipeの手の接続（21点）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 親指
    (0, 5), (5, 6), (6, 7), (7, 8),       # 人差し指
    (5, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),# 薬指
    (13, 17), (17, 18), (18, 19), (19, 20), # 小指
    (0, 17) # 手首と小指基部
]

st.set_page_config(page_title="3D Hand Distance Viewer", layout="centered")
st.title("📐 Thumb-Index 3D Distance Viewer")

uploaded_file = st.file_uploader("📄 Upload Hand Landmark CSV (21 landmarks x/y/z)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if df.shape[0] != 21 or not all(col in df.columns for col in ['x', 'y', 'z']):
        st.error("CSV must contain 21 rows and x/y/z columns.")
    else:
        coords = df[['x', 'y', 'z']].to_numpy()

        # 親指と人差し指の座標
        thumb_tip = coords[4]   # 親指の先
        index_tip = coords[8]   # 人差し指の先

        # 距離の計算
        distance = np.linalg.norm(index_tip - thumb_tip)

        # Y軸・Z軸を反転（視覚的整合性のため）
        xs, ys, zs = coords[:, 0], -coords[:, 1], -coords[:, 2]

        # --- 3D 描画 ---
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, zs, ys, c='blue', s=40)

        # 骨格を線で結ぶ
        for p1, p2 in HAND_CONNECTIONS:
            ax.plot([xs[p1], xs[p2]], [zs[p1], zs[p2]], [ys[p1], ys[p2]], 'gray')

        # ベクトル（親指→人差し指）を追加
        ax.quiver(
            thumb_tip[0], -thumb_tip[2], -thumb_tip[1],   # 始点 (X, Z, Y)
            index_tip[0] - thumb_tip[0],
            -(index_tip[2] - thumb_tip[2]),
            -(index_tip[1] - thumb_tip[1]),
            color='red', linewidth=2
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Z (Depth)")
        ax.set_zlabel("Y (Height)")
        ax.view_init(elev=15, azim=60)
        st.pyplot(fig)

        st.markdown(f"### 📏 Distance between Thumb & Index Tip: `{distance:.3f}` units")

        # CSV再表示とダウンロード
        st.markdown("### 🔢 Coordinates Table")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, file_name="hand_landmarks.csv", mime="text/csv")
