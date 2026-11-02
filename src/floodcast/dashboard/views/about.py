# src/floodcast/dashboard/views/about.py

"""Page 4: About — blue palette."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("About FloodCast BD")
    st.caption("Physics-informed flood prediction for Bangladesh")

    st.markdown("""
    <div class="card card-accent">
        <h4 style="color:#1b3554; margin-top:0;">What is this?</h4>
        <p style="color:#3f6593; line-height:1.7;">
        FloodCast BD predicts the <strong>chance of flooding</strong>
        at the Jamuna River in Sirajganj, Bangladesh. It looks at
        rainfall, river water levels, and soil moisture from the past
        25 years to make predictions for the next 3 days.
        </p>
        <p style="color:#3f6593; line-height:1.7;">
        What makes it special: the AI model follows
        <strong>physics rules</strong> — it understands that more rain
        and higher rivers mean more flooding, so its predictions
        are more realistic.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4 style="color:#1b3554; margin-top:0;">How it works</h4>
        <ol style="color:#3f6593; line-height:2.0; padding-left:20px;">
            <li>Download daily weather and river data from satellites
                </li>
            <li>Calculate useful numbers like "total rain in last
                7 days"</li>
            <li>Feed the numbers into AI models trained on 25 years
                of flood history</li>
            <li>The model gives a percentage chance of flooding</li>
            <li>Physics rules double-check the prediction makes
                sense</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4 style="color:#1b3554; margin-top:0;">Data sources</h4>
        <table style="width:100%; border-collapse:collapse;
            color:#3f6593;">
        <tr style="border-bottom:1px solid #c0e6fd;">
            <td style="padding:10px;">
                <strong style="color:#1b3554;">ERA5-Land</strong></td>
            <td style="padding:10px;">Weather data from the European
                weather centre (ECMWF)</td>
        </tr>
        <tr style="border-bottom:1px solid #c0e6fd;">
            <td style="padding:10px;">
                <strong style="color:#1b3554;">CHIRPS</strong></td>
            <td style="padding:10px;">Satellite-based rainfall
                measurements</td>
        </tr>
        <tr>
            <td style="padding:10px;">
                <strong style="color:#1b3554;">GloFAS</strong></td>
            <td style="padding:10px;">Global river water level
                data</td>
        </tr>
        </table>
        <p style="color:#5b86b6; font-size:13px; margin:10px 0 0 0;">
        Area covered: 24.0°N–25.5°N, 89.0°E–90.5°E
        (Sirajganj district) · Years: 2000–2025
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="border-left:4px solid #e65100;">
        <h4 style="color:#e65100; margin-top:0;">Important</h4>
        <ul style="color:#3f6593; line-height:1.8;">
            <li>This is a <strong>research project</strong>, not an
                official warning system</li>
            <li>Do NOT rely on this for real emergency decisions</li>
            <li>Always follow FFWC and local authority instructions
                </li>
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
        BSc Computer Science, AIUB
        """)
