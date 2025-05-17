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
import mediapipe as mp

st.set_page_config(page_title="3D Hand Viewer", layout="centered")
st.title("🖐️ MediaPipe Hand Skeleton + Thumb-Index Distance")

uploaded_file = st.file_uploader("📄 Upload a single-frame CSV (21 landmarks)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if df.shape[0] != 21 or not all(col in df.columns for col in ['x', 'y', 'z']):
        st.error("❌ Invalid CSV. Must have 21 rows and x/y/z columns.")
    else:
        thumb = df.iloc[4][['x', 'y', 'z']].astype(float).values
        index = df.iloc[8][['x', 'y', 'z']].astype(float).values
        distance = np.linalg.norm(thumb - index)
        st.success(f"📏 3D Distance (Thumb ↔ Index): **{distance:.4f} units**")

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Hand Landmarks with Skeleton & Distance Vector", pad=20)

        # 座標反転：MediaPipe座標系 → 直感的表示（上下・奥行き反転）
        df['y'] = -df['y']
        df['z'] = -df['z']

        # ランドマーク点
        ax.scatter(df['x'], df['z'], df['y'], color='blue', s=50, label='Landmarks')

        # 骨格線
        mp_hands = mp.solutions.hands
        for connection in mp_hands.HAND_CONNECTIONS:
            p1, p2 = connection
            x = [df.iloc[p1]['x'], df.iloc[p2]['x']]
            y = [df.iloc[p1]['z'], df.iloc[p2]['z']]
            z = [df.iloc[p1]['y'], df.iloc[p2]['y']]
            ax.plot(x, y, z, color='gray', linewidth=2)

        # 親指→人差し指の距離ベクトル
        ax.quiver(
            thumb[0], -thumb[2], -thumb[1],
            index[0] - thumb[0],
            -(index[2] - thumb[2]),
            -(index[1] - thumb[1]),
            color='red', arrow_length_ratio=0.1, linewidth=2
        )
        ax.text(thumb[0], -thumb[2], -thumb[1], "Thumb", color='red')
        ax.text(index[0], -index[2], -index[1], "Index", color='green')

        ax.set_xlabel("X")
        ax.set_ylabel("Z (Depth)")
        ax.set_zlabel("Y (Up)")
        ax.view_init(elev=20, azim=60)
        st.pyplot(fig)

        st.markdown("### 🧮 Coordinates")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="hand_landmarks.csv", mime="text/csv")