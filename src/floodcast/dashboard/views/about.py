# src/floodcast/dashboard/views/about.py

"""About page — Slate + Blue theme, updated for ERA5-only pipeline."""

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
        years of ERA5-Land reanalysis data (runoff, temperature,
        soil moisture, wind) to forecast floods up to 3 days ahead.
        </p>
        <p style="color:#cbd5e1; line-height:1.7;">
        The AI model follows
        <strong style="color:#f1f5f9;">physics rules</strong> —
        it knows more runoff and saturated soil mean higher flood
        risk, so predictions stay realistic.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4 style="color:#f1f5f9; margin-top:0;">How it works</h4>
        <ol style="color:#cbd5e1; line-height:2.0;
            padding-left:20px;">
            <li>Download daily ERA5-Land reanalysis data from ECMWF CDS</li>
            <li>Calculate features: 7-day runoff sum, soil saturation,
                wind magnitude</li>
            <li>Define flood label: runoff &gt; 90th percentile</li>
            <li>Train 3 ML models on 2000–2022 data, test on 2023–2025</li>
            <li>Shift labels by 1/2/3 days for lead-time forecasting</li>
            <li>Physics constraints enforce monotonicity (more runoff →
                higher risk)</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4 style="color:#f1f5f9; margin-top:0;">Data source</h4>
        <table style="width:100%; border-collapse:collapse;">
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:10px;">
                <strong style="color:#3b82f6;">ERA5-Land</strong></td>
            <td style="padding:10px; color:#cbd5e1;">
                Hourly reanalysis — runoff, temperature, soil moisture,
                wind (ECMWF)</td>
        </tr>
        </table>
        <p style="color:#64748b; font-size:12px; margin:10px 0 0 0;">
        24.0°N–25.5°N, 89.0°E–90.5°E (Sirajganj) · 2000–2025
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4 style="color:#f1f5f9; margin-top:0;">Variables</h4>
        <table style="width:100%; border-collapse:collapse;">
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; color:#3b82f6; font-weight:600;">
                runoff</td>
            <td style="padding:8px; color:#cbd5e1;">
                Surface + subsurface runoff (m/day)</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; color:#3b82f6; font-weight:600;">
                temperature_2m</td>
            <td style="padding:8px; color:#cbd5e1;">
                Air temperature at 2m height (K)</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; color:#3b82f6; font-weight:600;">
                soil_moisture L1/L2</td>
            <td style="padding:8px; color:#cbd5e1;">
                Volumetric soil water in top layers (m³/m³)</td>
        </tr>
        <tr>
            <td style="padding:8px; color:#3b82f6; font-weight:600;">
                wind u/v 10m</td>
            <td style="padding:8px; color:#cbd5e1;">
                Wind components at 10m height (m/s)</td>
        </tr>
        </table>
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
        - [ERA5-Land (CDS)](https://cds.climate.copernicus.eu/)
        - [FFWC Bangladesh](http://www.ffwc.gov.bd/)
        """)
    with col2:
        st.markdown("""
        **Built by**

        Md. Estiak Rahman Ayon
        (BSc Computer Science, AIUB)
        """)
