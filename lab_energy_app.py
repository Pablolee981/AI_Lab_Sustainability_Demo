import os
import requests

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL_PATH = "lab_model.pkl"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"


@st.cache_resource
def load_models():
    data = joblib.load(MODEL_PATH)
    return data["kwh_model"], data["score_model"], data["feature_columns"]


# ---------------------------------------------------------
# CLIMATE HELPERS – jak w oryginale
# ---------------------------------------------------------

def infer_climate_factor(city_name: str) -> str:
    """Fallback heuristic: city name -> Cool / Warm / Hot."""
    if not city_name:
        return "Warm"

    name = city_name.lower()

    hot_keywords = ["dubai", "riyadh", "cairo", "doha", "abu dhabi", "jeddah"]
    cool_keywords = ["oslo", "helsinki", "stockholm", "reykjavik", "alps", "zermatt"]
    warm_keywords = ["warsaw", "vienna", "berlin", "paris", "rome", "madrid", "london"]

    if any(k in name for k in hot_keywords):
        return "Hot"
    if any(k in name for k in cool_keywords):
        return "Cool"
    if any(k in name for k in warm_keywords):
        return "Warm"

    return "Warm"


def fetch_climate_from_city(city_name: str):
    """
    OpenWeatherMap → temperatura → kategoria klimatu: Cool / Warm / Hot.

    Zwraca (climate_factor, temperature_celsius, api_used_successfully).
    """
    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key or not city_name:
        return None, None, False

    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric",
    }

    try:
        resp = requests.get(WEATHER_API_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        temp_c = data["main"]["temp"]

        if temp_c < 5:
            climate_factor = "Cool"
        elif temp_c < 22:
            climate_factor = "Warm"
        else:
            climate_factor = "Hot"

        return climate_factor, temp_c, True
    except Exception:
        return None, None, False


# ---------------------------------------------------------
# MODEL INPUTS
# ---------------------------------------------------------

def build_feature_vector(
    width_cm,
    length_cm,
    height_cm,
    num_computers,
    hours_per_day,
    lighting_type,
    has_ac,
    climate_factor,
    feature_columns,
):
    """Buduje wektor cech dokładnie tak jak w treningu (bez nowych pól)."""
    base = {
        "width_cm": width_cm,
        "length_cm": length_cm,
        "height_cm": height_cm,
        "num_computers": num_computers,
        "hours_per_day": hours_per_day,
        "has_ac": 1 if has_ac else 0,
        "lighting_type": lighting_type,
        "climate_factor": climate_factor,
    }

    df = pd.DataFrame([base])
    df = pd.get_dummies(df, columns=["lighting_type", "climate_factor"], drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df


# ---------------------------------------------------------
# REKOMENDACJE I WYJAŚNIENIA – niezmienione
# ---------------------------------------------------------

def generate_recommendations(
    monthly_kwh,
    user_overridden_kwh,
    sustainability_score,
    volume_m3,
    num_computers,
    hours_per_day,
    lighting_type,
    has_ac,
    climate_factor,
):
    recs = []

    effective_kwh = user_overridden_kwh if user_overridden_kwh is not None else monthly_kwh
    kwh_per_m3 = effective_kwh / max(volume_m3, 1)

    effective_score_from_energy = 100 - np.clip(kwh_per_m3 * 2, 0, 90)
    combined_score = float(
        np.clip((sustainability_score + effective_score_from_energy) / 2.0, 0, 100)
    )

    if combined_score >= 75:
        recs.append(
            "The lab is already performing well in terms of energy efficiency. Focus on maintaining current good practices."
        )
    elif combined_score >= 50:
        recs.append(
            "The lab has moderate sustainability. There is clear potential to reduce energy use further."
        )
    else:
        recs.append(
            "The lab shows high energy intensity. Consider more aggressive measures to improve efficiency."
        )

    if num_computers > 15 and hours_per_day > 8:
        recs.append(
            "Introduce automatic shutdown or sleep policies for computers outside teaching hours."
        )
    elif hours_per_day > 10:
        recs.append(
            "Review computer usage schedule and reduce idle time where possible."
        )

    if lighting_type == "Fluorescent":
        recs.append(
            "Consider replacing fluorescent lighting with LED fixtures to reduce lighting consumption."
        )
    elif lighting_type == "Other":
        recs.append(
            "Evaluate the efficiency of the current lighting system and consider migrating to LED where feasible."
        )

    if has_ac:
        if climate_factor == "Hot":
            recs.append(
                "Optimize air conditioning settings (temperature setpoints, scheduling) to avoid overcooling in a hot climate."
            )
        elif climate_factor == "Warm":
            recs.append(
                "Use occupancy-based control for air conditioning to avoid cooling empty rooms."
            )
        else:
            recs.append(
                "Check whether air conditioning is required at all times; consider using natural ventilation when outdoor conditions allow."
            )
    else:
        if climate_factor == "Hot":
            recs.append(
                "In a hot climate without AC, consider shading, reflective materials, or improved natural ventilation to reduce heat gains."
            )

    if kwh_per_m3 > 15:
        recs.append(
            "Energy use per cubic meter is high; review overall operating hours, standby settings, and equipment efficiency."
        )

    if not recs:
        recs.append(
            "No specific issues detected. Monitor energy use over time to spot new optimization opportunities."
        )

    return recs


def generate_recommendations_llm(
    monthly_kwh,
    user_overridden_kwh,
    sustainability_score,
    volume_m3,
    num_computers,
    hours_per_day,
    lighting_type,
    has_ac,
    climate_factor,
):
    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise RuntimeError("LLM API key not configured (LLM_API_KEY).")

    # W demie używasz tych samych reguł – zostawiamy to bez zmian.
    simulated_recs = generate_recommendations(
        monthly_kwh=monthly_kwh,
        user_overridden_kwh=user_overridden_kwh,
        sustainability_score=sustainability_score,
        volume_m3=volume_m3,
        num_computers=num_computers,
        hours_per_day=hours_per_day,
        lighting_type=lighting_type,
        has_ac=has_ac,
        climate_factor=climate_factor,
    )

    return simulated_recs


def build_explanation(
    sustainability_score,
    monthly_kwh,
    volume_m3,
    climate_factor,
    lighting_type,
    num_computers,
    hours_per_day,
):
    effective_density = monthly_kwh / max(volume_m3, 1)
    explanation = []

    explanation.append(
        f"Sustainability score is based on estimated energy use per cubic meter (currently about {effective_density:.1f} kWh/m³ per month)."
    )

    explanation.append(
        f"The model also considers the number of computers ({num_computers}) and their average operating hours ({hours_per_day:.1f} h/day)."
    )

    explanation.append(
        f"Lighting type ({lighting_type}) and climate conditions ({climate_factor}) have additional impact on the final score."
    )

    explanation.append(
        "Lower energy use per volume, efficient lighting (LED), and moderate climates lead to higher sustainability scores."
    )

    return explanation


# ---------------------------------------------------------
# GŁÓWNA APLIKACJA – nowy layout + stare zachowanie
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AI-based Computer Lab Sustainability Demo",
        layout="wide",
    )

    st.title("AI-based Computer Lab Sustainability Demo")

    st.markdown(
        """
This demo estimates monthly energy use and a sustainability score for a computer lab,
based on room size, equipment, usage patterns, and climate.

*Note: The app first tries to fetch climate data from OpenWeatherMap using the city name.
If this fails (e.g. typos, API limits), it falls back to a simple template-based climate category.*
"""
    )

    kwh_model, score_model, feature_columns = load_models()

    # ---------- GLOBAL OPTIONS ----------
    with st.sidebar:
        st.header("Global options")
        use_llm = st.checkbox(
            "Use AI-generated recommendations (LLM)",
            value=False,
            help="If enabled, the app will try to use an external LLM API. "
                 "In this demo, if no API key is configured, it will fall back to rule-based templates.",
        )

        use_override = st.checkbox(
            "Use overridden monthly energy value for recommendations",
            value=False,
            help="If enabled, the user-provided kWh value will be used when generating recommendations.",
        )

    st.markdown("---")
    st.subheader("Lab characteristics")

    col1, col2 = st.columns(2)
    with col1:
        width_cm = st.number_input("Room width (cm)", min_value=300, max_value=2000, value=800, step=10)
        length_cm = st.number_input("Room length (cm)", min_value=300, max_value=2500, value=1000, step=10)
        height_cm = st.number_input("Room height (cm)", min_value=240, max_value=400, value=300, step=5)
        num_computers = st.number_input("Number of computers", min_value=1, max_value=100, value=20, step=1)
    with col2:
        hours_per_day = st.number_input(
            "Average computer usage (hours/day)", min_value=1.0, max_value=24.0, value=8.0, step=0.5
        )
        lighting_type = st.selectbox("Lighting type", ["LED", "Fluorescent", "Other"])
        has_ac = st.checkbox("Air conditioning present", value=True)
        city = st.text_input("City or location (for climate data)", value="Warsaw, PL")

    # ---------- NOWE POLA – TYLKO INFORMACYJNE ----------
    st.markdown("---")
    st.subheader("Additional room & envelope details (informational)")

    col3, col4, col5 = st.columns(3)
    with col3:
        room_area_m2 = (width_cm / 100) * (length_cm / 100)
        st.metric("Calculated floor area [m²]", f"{room_area_m2:.1f}")
        window_area = st.number_input(
            "Total window area [m²]",
            min_value=0.0,
            value=10.0,
            step=0.5,
        )

    with col4:
        window_orientation = st.selectbox(
            "Window orientation",
            options=[
                "North",
                "East",
                "South",
                "West",
                "North-East",
                "South-East",
                "South-West",
                "North-West",
                "Mixed / not sure",
            ],
            index=2,
        )
        cooling_capacity = st.number_input(
            "Cooling capacity of air conditioning [kW]",
            min_value=0.0,
            value=5.0,
            step=0.5,
        )

    with col5:
        power_supply_type = st.selectbox(
            "Power supply type",
            options=[
                "Grid only",
                "On-site PV only",
                "Mixed (grid + PV)",
            ],
            index=0,
        )
        smart_power_mgmt = st.selectbox(
            "Smart power management enabled?",
            options=["Yes", "No", "Partially / not sure"],
            index=2,
        )

    st.markdown("---")
    st.subheader("IT & AV equipment (informational)")

    col6, col7, col8 = st.columns(3)
    with col6:
        num_workstations = st.number_input(
            "Number of computer workstations",
            min_value=0,
            value=20,
            step=1,
        )
    with col7:
        total_power_per_workstation = st.number_input(
            "Total power per workstation [W]",
            min_value=0,
            value=150,
            step=10,
        )
    with col8:
        av_power = st.number_input(
            "Additional AV equipment power [W]",
            min_value=0,
            value=300,
            step=50,
        )

    computing_setup_type = st.selectbox(
        "Computing setup type",
        options=[
            "Local high-performance PCs",
            "Standard office PCs",
            "Thin clients with server-side processing",
            "Mixed / not sure",
        ],
        index=1,
    )

    st.markdown("---")
    st.subheader("User-provided energy (optional)")

    user_monthly_kwh_input = st.number_input(
        "Monthly energy use (kWh) (user input, optional)",
        min_value=0.0,
        value=0.0,
        step=10.0,
        help="If you know the actual monthly energy consumption, enter it here. "
             "It will be used for recommendations only when the override checkbox in the sidebar is enabled.",
    )

    # ---------- POGODA / KLIMAT ----------
    api_climate_factor, temp_c, api_ok = fetch_climate_from_city(city)

    if api_ok:
        climate_factor = api_climate_factor
        st.success(
            f"Weather API: current temperature in **{city}** is about **{temp_c:.1f} °C** "
            f"→ climate category: **{climate_factor}**."
        )
    else:
        climate_factor = infer_climate_factor(city)
        st.warning(
            "Could not retrieve climate data from the weather API (missing key, typo, or network issue). "
            f"Using template-based climate category instead: **{climate_factor}**."
        )

    volume_m3 = (width_cm / 100) * (length_cm / 100) * (height_cm / 100)

    # ---------- PRZYCISK I CAŁY STARY FLOW ----------
    if st.button("Estimate energy use and sustainability score") or st.session_state.get(
        "has_predictions", False
    ):
        st.session_state["has_predictions"] = True

        X = build_feature_vector(
            width_cm,
            length_cm,
            height_cm,
            num_computers,
            hours_per_day,
            lighting_type,
            has_ac,
            climate_factor,
            feature_columns,
        )

        predicted_monthly_kwh = float(kwh_model.predict(X)[0])
        predicted_score = float(score_model.predict(X)[0])

        st.subheader("Model predictions")
        st.write(f"**Predicted monthly energy use (all sources):** {predicted_monthly_kwh:.1f} kWh")
        st.write(f"**Predicted sustainability score:** {predicted_score:.1f} / 100")

        if use_override and user_monthly_kwh_input > 0:
            effective_kwh = user_monthly_kwh_input
            st.info(
                "Recommendations and explanations are based on the **user-provided** monthly energy use value."
            )
        else:
            effective_kwh = predicted_monthly_kwh
            st.info(
                "Recommendations and explanations are based on the **predicted** monthly energy use value."
            )

        st.markdown("---")
        st.subheader("Recommendations")

        if use_llm:
            try:
                recs = generate_recommendations_llm(
                    monthly_kwh=predicted_monthly_kwh,
                    user_overridden_kwh=effective_kwh,
                    sustainability_score=predicted_score,
                    volume_m3=volume_m3,
                    num_computers=num_computers,
                    hours_per_day=hours_per_day,
                    lighting_type=lighting_type,
                    has_ac=has_ac,
                    climate_factor=climate_factor,
                )
                st.info(
                    "Recommendations below are generated via the LLM hook. "
                    "In this demo they reuse rule-based templates unless an external API key is configured."
                )
            except Exception as e:
                st.warning(
                    "LLM-based recommendations are not available "
                    f"(reason: {e}). Recommendations below are generated from rule-based templates."
                )
                recs = generate_recommendations(
                    monthly_kwh=predicted_monthly_kwh,
                    user_overridden_kwh=effective_kwh,
                    sustainability_score=predicted_score,
                    volume_m3=volume_m3,
                    num_computers=num_computers,
                    hours_per_day=hours_per_day,
                    lighting_type=lighting_type,
                    has_ac=has_ac,
                    climate_factor=climate_factor,
                )
        else:
            st.info(
                "Recommendations below are generated from rule-based templates "
                "(LLM option is disabled in the sidebar)."
            )
            recs = generate_recommendations(
                monthly_kwh=predicted_monthly_kwh,
                user_overridden_kwh=effective_kwh,
                sustainability_score=predicted_score,
                volume_m3=volume_m3,
                num_computers=num_computers,
                hours_per_day=hours_per_day,
                lighting_type=lighting_type,
                has_ac=has_ac,
                climate_factor=climate_factor,
            )

        for r in recs:
            st.markdown(f"- {r}")

        st.markdown("---")
        st.subheader("Why this score?")

        explanation_lines = build_explanation(
            sustainability_score=predicted_score,
            monthly_kwh=effective_kwh,
            volume_m3=volume_m3,
            climate_factor=climate_factor,
            lighting_type=lighting_type,
            num_computers=num_computers,
            hours_per_day=hours_per_day,
        )

        for line in explanation_lines:
            st.write(line)


if __name__ == "__main__":
    main()
