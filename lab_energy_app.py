import streamlit as st

st.set_page_config(page_title="AI-based Sustainability Lab Demo", layout="wide")

st.title("AI-based Sustainability Lab Demo")

with st.sidebar:
    st.header("Global options")
    use_llm = st.checkbox("Use AI-generated recommendations (LLM)", value=False)
    override_energy = st.checkbox("Override with measured monthly energy use (kWh)", value=False)
    measured_energy = None
    if override_energy:
        measured_energy = st.number_input(
            "Measured monthly energy use [kWh]",
            min_value=0.0,
            value=500.0,
            step=10.0,
        )

st.header("Room & envelope")

col1, col2, col3 = st.columns(3)

with col1:
    room_area = st.number_input(
        "Room floor area [m²]",
        min_value=5.0,
        value=60.0,
        step=1.0,
    )
    room_height = st.number_input(
        "Room height [m]",
        min_value=2.5,
        value=3.0,
        step=0.1,
    )

with col2:
    window_area = st.number_input(
        "Total window area [m²]",
        min_value=0.0,
        value=10.0,
        step=0.5,
    )
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

with col3:
    hvac_present = st.selectbox(
        "Air conditioning present?",
        options=["Yes", "No"],
        index=0,
    )
    cooling_capacity = st.number_input(
        "Cooling capacity of air conditioning [kW]",
        min_value=0.0,
        value=5.0,
        step=0.5,
    )

st.divider()
st.header("IT & AV equipment")

col4, col5, col6 = st.columns(3)

with col4:
    num_computers = st.number_input(
        "Number of computers / terminals",
        min_value=0,
        value=20,
        step=1,
    )
    num_workstations = st.number_input(
        "Number of computer workstations",
        min_value=0,
        value=20,
        step=1,
        help="Number of seats with a computer or thin client available to users.",
    )

with col5:
    total_power_per_workstation = st.number_input(
        "Total power per workstation [W]",
        min_value=0,
        value=150,
        step=10,
        help="Approximate power of PC + monitor or thin client + monitor.",
    )
    av_power = st.number_input(
        "Additional AV equipment power [W]",
        min_value=0,
        value=300,
        step=50,
        help="Projectors, additional displays, speakers, etc.",
    )

with col6:
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
st.header("Operation & context")

col7, col8, col9 = st.columns(3)

with col7:
    hours_per_day = st.number_input(
        "Average hours of use per day",
        min_value=1.0,
        value=8.0,
        step=0.5,
    )
    days_per_month = st.number_input(
        "Days of use per month",
        min_value=1,
        value=22,
        step=1,
    )

with col8:
    lighting_type = st.selectbox(
        "Lighting type",
        options=["LED", "Fluorescent", "Halogen / other"],
        index=0,
    )
    typical_occupancy = st.selectbox(
        "Typical student occupancy",
        options=["Low", "Medium", "High"],
        index=1,
        help="Rough indication of how full the room usually is during operation.",
    )

with col9:
    power_supply_type = st.selectbox(
        "Power supply type",
        options=[
            "Grid only",
            "On-site PV only",
            "Mixed (grid + PV)",
        ],
        index=0,
        help="This is informational for now and does not change calculations.",
    )
    smart_power_mgmt = st.selectbox(
        "Smart power management enabled?",
        options=["Yes", "No", "Partially / not sure"],
        index=2,
        help="For example automatic sleep modes, screen off timers, etc.",
    )

st.divider()

st.subheader("Location & climate")

col10, col11 = st.columns(2)

with col10:
    city_name = st.text_input(
        "City (for climate lookup)",
        value="Warsaw",
    )

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

st.divider()
st.subheader("Run assessment")

run_button = st.button("Estimate energy use and sustainability score")

# --- existing model + logic below ---
# Use room_area, room_height, num_computers, hours_per_day, days_per_month,
# hvac_present, lighting_type, etc. for the current model.
# Newly added fields (num_workstations, total_power_per_workstation,
# cooling_capacity, window_area, window_orientation, av_power, computer_type,
# typical_occupancy, power_supply_type, smart_power_mgmt) are collected
# but not yet used in calculations.
