import math
import os
import requests
import streamlit as st

# ---------- CONFIG ----------

st.set_page_config(
    page_title="AI-based Computer Lab Sustainability Demo",
    layout="wide"
)

# ---------- TITLE & INTRO ----------

st.title("AI-based Computer Lab Sustainability Demo")

st.write(
    """
This demo estimates monthly energy use and a sustainability score for a computer lab, 
based on room size, equipment, usage patterns, and climate.
"""
)

st.caption(
    """
Note: The app first tries to fetch climate data from OpenWeatherMap using the city name. 
If this fails (e.g. typos, API limits), it falls back to a simple template-based climate category.
"""
)

# ---------- GLOBAL OPTIONS (SIDEBAR) ----------

with st.sidebar:
    st.header("Global options")
    use_llm = st.checkbox("Use AI-generated recommendations (LLM)", value=False)
    override_energy = st.checkbox(
        "Use overridden monthly energy value for recommendations",
        value=False,
        help="If enabled, recommendations will be based on the value you provide below "
             "instead of the model-predicted monthly energy use."
    )
    measured_energy = None
    if override_energy:
        measured_energy = st.number_input(
            "Monthly energy use (kWh) (user input, optional)",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )

# ---------- LAB CHARACTERISTICS (OLD + NEW FIELDS) ----------

st.subheader("Lab characteristics")

col1, col2 = st.columns(2)

with col1:
    room_width_cm = st.number_input(
        "Room width (cm)", min_value=100, value=800, step=10
    )
    room_length_cm = st.number_input(
        "Room length (cm)", min_value=100, value=1000, step=10
    )
    room_height_cm = st.number_input(
        "Room height (cm)", min_value=200, value=300, step=10
    )
    num_computers = st.number_input(
        "Number of computers", min_value=0, value=20, step=1
    )

with col2:
    avg_hours = st.number_input(
        "Average computer usage (hours/day)",
        min_value=1.0,
        value=8.0,
        step=0.5,
    )
    lighting_type = st.selectbox(
        "Lighting type",
        options=["LED", "Fluorescent", "Halogen / other"],
        index=0,
    )
    hvac_present = st.checkbox("Air conditioning present", value=True)
    city_name = st.text_input(
        "City or location (for climate data)",
        value="Warsaw, PL",
    )

st.divider()
st.subheader("Additional room & envelope details (informational)")

col3, col4, col5 = st.columns(3)

with col3:
    room_area_m2 = (room_width_cm / 100) * (room_length_cm / 100)
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
        help="Informational only – does not change calculations yet.",
    )
    smart_power_mgmt = st.selectbox(
        "Smart power management enabled?",
        options=["Yes", "No", "Partially / not sure"],
        index=2,
    )

st.divider()
st.subheader("IT & AV equipment (informational additions)")

col6, col7, col8 = st.columns(3)

with col6:
    num_workstations = st.number_input(
        "Number of computer workstations",
        min_value=0,
        value=20,
        step=1,
        help="Number of seats with a computer or thin client.",
    )

with col7:
    total_power_per_workstation = st.number_input(
        "Total power per workstation [W]",
        min_value=0,
        value=150,
        step=10,
        help="Approximate power of PC + monitor or thin client + monitor.",
    )

with col8:
    av_power = st.number_input(
        "Additional AV equipment power [W]",
        min_value=0,
        value=300,
        step=50,
        help="Projectors, additional displays, speakers, etc.",
    )

computer_type = st.selectbox(
    "Computing setup type",
    options=[
        "Local high-performance PCs",
        "Standard office PCs",
        "Thin clients with server-side processing",
        "Mixed / not sure",
    ],
    index=1,
    help="Whether most processing happens on local desktops or in a data centre.",
)

st.divider()
st.subheader("Operation & occupancy")

col9, col10 = st.columns(2)

with col9:
    days_per_month = st.number_input(
        "Days of use per month",
        min_value=1,
        value=22,
        step=1,
    )

with col10:
    typical_occupancy = st.selectbox(
        "Typical student occupancy",
        options=["Low", "Medium", "High"],
        index=1,
    )

st.divider()
st.subheader("Climate options")

col11, col12 = st.columns(2)

with col11:
    climate_override = st.selectbox(
        "Override climate category?",
        options=[
            "Use weather API (if available)",
            "Cool",
            "Moderate",
            "Warm",
            "Hot",
        ],
        index=0,
    )

with col12:
    st.write("")

# ---------- WEATHER & CLIMATE HELPERS ----------

def classify_climate_from_temp(temp_c: float) -> str:
    if temp_c <= 5:
        return "Cool"
    elif temp_c <= 18:
        return "Moderate"
    elif temp_c <= 28:
        return "Warm"
    else:
        return "Hot"

def get_weather_and_climate(city: str, override: str):
    """
    Returns (temperature_C or None, climate_category, message_string).
    """
    api_mode = "API"
    temp_c = None
    climate = None

    if override != "Use weather API (if available)":
        climate = override
        api_mode = "Override"
    else:
        api_key = os.getenv("WEATHER_API_KEY")
        if api_key:
            try:
                url = (
                    "https://api.openweathermap.org/data/2.5/weather"
                    f"?q={city}&appid={api_key}&units=metric"
                )
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    temp_c = data["main"]["temp"]
                    climate = classify_climate_from_temp(temp_c)
                else:
                    api_mode = f"API error {resp.status_code}"
            except Exception as e:
                api_mode = f"API error: {e}"
        else:
            api_mode = "No API key"

    if climate is None:
        # Fallback template if API and override both fail somehow
        climate = "Moderate"

    if api_mode == "API":
        msg = (
            f"Weather API: current temperature in {city} is about {temp_c:.1f} °C "
            f"→ climate category: {climate}."
        )
    elif api_mode == "Override":
        msg = f"Climate category overridden to: {climate}."
    elif api_mode == "No API key":
        msg = (
            f"No WEATHER_API_KEY found – falling back to default climate category: {climate}."
        )
    else:
        msg = (
            f"Weather API issue ({api_mode}) – falling back to default climate "
            f"category: {climate}."
        )

    return temp_c, climate, msg

# ---------- SIMPLE ENERGY MODEL (AS BEFORE) ----------

def estimate_monthly_energy_kwh(
    room_width_cm,
    room_length_cm,
    room_height_cm,
    num_computers,
    avg_hours,
    days_per_month,
    hvac_present,
    lighting_type,
    climate_category,
):
    # Room volume
    volume_m3 = (room_width_cm / 100) * (room_length_cm / 100) * (room_height_cm / 100)

    # Base IT load (rough heuristic)
    if num_computers <= 0:
        base_it_kw = 0.0
    else:
        base_it_kw = num_computers * 0.12  # 120 W per computer as rough average

    # Lighting power density
    if lighting_type == "LED":
        lpd_w_m2 = 6
    elif lighting_type == "Fluorescent":
        lpd_w_m2 = 10
    else:
        lpd_w_m2 = 14

    floor_area_m2 = (room_width_cm / 100) * (room_length_cm / 100)
    lighting_kw = (lpd_w_m2 * floor_area_m2) / 1000.0

    # HVAC factor based on climate + presence
    if not hvac_present:
        hvac_kw = 0.0
    else:
        if climate_category == "Cool":
            hvac_kw = 0.5
        elif climate_category == "Moderate":
            hvac_kw = 0.8
        elif climate_category == "Warm":
            hvac_kw = 1.2
        else:
            hvac_kw = 1.6

    total_kw = base_it_kw + lighting_kw + hvac_kw

    monthly_kwh = total_kw * avg_hours * days_per_month
    return monthly_kwh

def sustainability_score_from_energy(kwh: float, floor_area_m2: float) -> int:
    # Simple intensity-based score (0–100)
    if floor_area_m2 <= 0:
        return 50
    intensity = kwh / floor_area_m2  # kWh/m2 per month

    # Piecewise mapping
    if intensity <= 8:
        score = 90
    elif intensity <= 15:
        score = 75
    elif intensity <= 25:
        score = 60
    elif intensity <= 40:
        score = 45
    else:
        score = 30
    return int(score)

# ---------- RUN BUTTON ----------

st.divider()
if st.button("Estimate energy use and sustainability score"):
    temp_c, climate_category, climate_msg = get_weather_and_climate(
        city_name, climate_override
    )

    st.success(climate_msg)

    monthly_kwh_model = estimate_monthly_energy_kwh(
        room_width_cm=room_width_cm,
        room_length_cm=room_length_cm,
        room_height_cm=room_height_cm,
        num_computers=num_computers,
        avg_hours=avg_hours,
        days_per_month=days_per_month,
        hvac_present=hvac_present,
        lighting_type=lighting_type,
        climate_category=climate_category,
    )

    # Energy used for scoring / recommendations
    if override_energy and measured_energy and measured_energy > 0:
        energy_for_score = measured_energy
        st.info(
            f"Using overridden monthly energy value for recommendations: "
            f"{measured_energy:.0f} kWh."
        )
    else:
        energy_for_score = monthly_kwh_model

    score = sustainability_score_from_energy(
        energy_for_score,
        floor_area_m2=room_area_m2,
    )

    st.subheader("Results")
    st.metric("Model-estimated monthly energy use [kWh]", f"{monthly_kwh_model:.0f}")
    if override_energy and measured_energy and measured_energy > 0:
        st.metric(
            "Energy value used for recommendations [kWh]",
            f"{energy_for_score:.0f}",
        )
    st.metric("Sustainability score (0–100)", f"{score}")

    # Tu możesz dalej wpiąć swoje rekomendacje, w tym ewentualne LLM.
    # ...
