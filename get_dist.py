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
from io import BytesIO

st.set_page_config(page_title="Hand Distance Analyzer", layout="centered")
st.title("🖐️ 3D Hand Pose & Thumb-Index Distance")

uploaded_file = st.file_uploader("📄 Upload a CSV with 3D landmarks (x, y, z)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    if all(col in df.columns for col in ['x', 'y', 'z']):
        coords = df[['x', 'y', 'z']].values.reshape(-1, 3)
        if coords.shape[0] != 21:
            st.error("Expected 21 landmarks (MediaPipe format).")
        else:
            thumb_tip = coords[4]
            index_tip = coords[8]
            distance = np.linalg.norm(thumb_tip - index_tip)
            st.markdown(f"### 📏 Distance: `{distance:.3f}` units")

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')

            xs, ys, zs = coords[:, 0], -coords[:, 1], -coords[:, 2]
            ax.scatter(xs, zs, ys, c='blue', s=40)

            HAND_CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20)
            ]
            for p1, p2 in HAND_CONNECTIONS:
                ax.plot([xs[p1], xs[p2]], [zs[p1], zs[p2]], [ys[p1], ys[p2]], 'gray')

            ax.quiver(
                thumb_tip[0], -thumb_tip[2], -thumb_tip[1],
                index_tip[0] - thumb_tip[0],
                -(index_tip[2] - thumb_tip[2]),
                -(index_tip[1] - thumb_tip[1]),
                color='red', linewidth=2
            )

            ax.set_xlabel("X")
            ax.set_ylabel("Z (Depth)")
            ax.set_zlabel("Y")
            ax.view_init(elev=10, azim=70)
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                "⬇️ Download Plot (PNG)",
                data=buf.getvalue(),
                file_name="hand_pose.png",
                mime="image/png"
            )
    else:
        st.error("The CSV must include columns: x, y, z.")
