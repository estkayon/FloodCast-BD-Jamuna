# src/floodcast/dashboard/views/about.py

"""About page — Slate + Blue theme."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("About FloodCast BD")
    st.caption("Physics-informed flood prediction for Bangladesh")

    st.markdown("""
    <div class="card card-accent">
        <h4 style="color:#f1f5f9; margin-top:0;">What is this?</h4>
        <p style="color:#cbd5e1; line-height:1.7;">
        FloodCast BD predicts the
        <strong style="color:#f1f5f9;">chance of flooding</strong>
        at the Jamuna River in Sirajganj, Bangladesh. It uses 25
        years of rainfall, river levels, and soil moisture data
        to forecast the next 3 days.
        </p>
        <p style="color:#cbd5e1; line-height:1.7;">
        The AI model follows
        <strong style="color:#f1f5f9;">physics rules</strong> —
        it knows more rain and higher rivers mean more flooding,
        so predictions stay realistic.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4 style="color:#f1f5f9; margin-top:0;">How it works</h4>
        <ol style="color:#cbd5e1; line-height:2.0;
            padding-left:20px;">
            <li>Download daily weather and river data</li>
            <li>Calculate features like "rain in last 7 days"</li>
            <li>Feed into AI models trained on 25 years of
                history</li>
            <li>Model outputs a flood probability</li>
            <li>Physics rules verify the prediction</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4 style="color:#f1f5f9; margin-top:0;">Data sources</h4>
        <table style="width:100%; border-collapse:collapse;">
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:10px;">
                <strong style="color:#3b82f6;">ERA5-Land</strong></td>
            <td style="padding:10px; color:#cbd5e1;">
                Weather data (ECMWF)</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:10px;">
                <strong style="color:#3b82f6;">CHIRPS</strong></td>
            <td style="padding:10px; color:#cbd5e1;">
                Satellite rainfall</td>
        </tr>
        <tr>
            <td style="padding:10px;">
                <strong style="color:#3b82f6;">GloFAS</strong></td>
            <td style="padding:10px; color:#cbd5e1;">
                River water levels</td>
        </tr>
        </table>
        <p style="color:#64748b; font-size:12px; margin:10px 0 0 0;">
        24.0°N–25.5°N, 89.0°E–90.5°E (Sirajganj) · 2000–2025
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="border-left:3px solid #ef4444;">
        <h4 style="color:#ef4444; margin-top:0;">Important</h4>
        <ul style="color:#cbd5e1; line-height:1.8;">
            <li><strong style="color:#f1f5f9;">Research project
                </strong>, not official warning</li>
            <li>Do NOT use for emergency decisions</li>
            <li>Follow FFWC and local authorities</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Links**

        - [GitHub](https://github.com/estkayon/FloodCast-BD-Jamuna)
        - [ERA5-Land](https://cds.climate.copernicus.eu/)
        - [CHIRPS](https://www.chc.ucsb.edu/data/chirps)
        - [GloFAS](https://global-flood.emergency.copernicus.eu/)
        """)
    with col2:
        st.markdown("""
        **Built by**

        Md. Estiak Rahman Ayon
        (BSc Computer Science, AIUB)
        """)
