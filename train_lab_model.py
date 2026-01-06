import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import random

# Ustawienie losowości dla powtarzalności
np.random.seed(42)
random.seed(42)


def generate_synthetic_labs(n_samples: int = 800) -> pd.DataFrame:
    rows = []
    for _ in range(n_samples):
        # Wymiary pomieszczenia (cm) – typowe laby
        width_cm = np.random.randint(400, 1200)      # 4–12 m
        length_cm = np.random.randint(500, 1500)     # 5–15 m
        height_cm = np.random.randint(260, 350)      # 2.6–3.5 m

        volume_m3 = (width_cm / 100) * (length_cm / 100) * (height_cm / 100)

        # Liczba komputerów i godziny pracy
        num_computers = np.random.randint(5, 40)
        hours_per_day = np.random.uniform(4, 16)

        # Oświetlenie
        lighting_type = random.choice(["LED", "Fluorescent", "Other"])
        if lighting_type == "LED":
            lighting_power_per_m2 = 5   # W/m2 (umowne)
        elif lighting_type == "Fluorescent":
            lighting_power_per_m2 = 10
        else:
            lighting_power_per_m2 = 8

        area_m2 = (width_cm / 100) * (length_cm / 100)

        # Klimatyzacja
        has_ac = random.choice([0, 1])

        # Klimat syntetyczny
        climate_factor = random.choice(["Cool", "Warm", "Hot"])

        # --- "Prawdziwe" zużycie energii (kWh/miesiąc) ---

        # Komputery – załóżmy 150 W na komputer
        pc_power_kw = 0.15 * num_computers
        pc_energy_kwh = pc_power_kw * hours_per_day * 30  # 30 dni/miesiąc

        # Oświetlenie
        lighting_power_kw = (lighting_power_per_m2 * area_m2) / 1000
        lighting_hours_per_day = max(hours_per_day, 6)
        lighting_energy_kwh = lighting_power_kw * lighting_hours_per_day * 30

        # Klimatyzacja – zależna od klimatu i AC
        if has_ac:
            if climate_factor == "Hot":
                ac_base_kw = 0.15 * area_m2 / 10
                ac_hours = 12
            elif climate_factor == "Warm":
                ac_base_kw = 0.10 * area_m2 / 10
                ac_hours = 8
            else:  # Cool
                ac_base_kw = 0.05 * area_m2 / 10
                ac_hours = 4
            ac_energy_kwh = ac_base_kw * ac_hours * 30
        else:
            ac_energy_kwh = 0.0

        base_monthly_kwh = pc_energy_kwh + lighting_energy_kwh + ac_energy_kwh

        noise = np.random.normal(0, base_monthly_kwh * 0.05)
        monthly_kwh = max(base_monthly_kwh + noise, 10)

        # Sustainability score 0–100
        kwh_per_m3 = monthly_kwh / max(volume_m3, 1)

        score_from_energy = 100 - np.clip(kwh_per_m3 * 2, 0, 80)

        if lighting_type == "LED":
            score_from_energy += 5
        elif lighting_type == "Fluorescent":
            score_from_energy -= 5

        if climate_factor == "Hot":
            score_from_energy -= 10
        elif climate_factor == "Warm":
            score_from_energy -= 3

        sustainability_score = float(np.clip(score_from_energy, 0, 100))

        rows.append(
            {
                "width_cm": width_cm,
                "length_cm": length_cm,
                "height_cm": height_cm,
                "num_computers": num_computers,
                "hours_per_day": hours_per_day,
                "lighting_type": lighting_type,
                "has_ac": has_ac,
                "climate_factor": climate_factor,
                "monthly_kwh": monthly_kwh,
                "sustainability_score": sustainability_score,
            }
        )

    return pd.DataFrame(rows)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=["lighting_type", "climate_factor"], drop_first=True)
    return df


def train_models():
    df = generate_synthetic_labs(800)

    df.to_csv("synthetic_labs.csv", index=False)

    df_encoded = encode_features(df)

    X = df_encoded.drop(columns=["monthly_kwh", "sustainability_score"])
    y_kwh = df_encoded["monthly_kwh"]
    y_score = df_encoded["sustainability_score"]

    X_train, X_test, yk_train, yk_test = train_test_split(
        X, y_kwh, test_size=0.2, random_state=42
    )
    _, _, ys_train, ys_test = train_test_split(
        X, y_score, test_size=0.2, random_state=42
    )

    # Lżejsze modele: mniej drzew
    kwh_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42
    )
    kwh_model.fit(X_train, yk_train)
    kwh_pred = kwh_model.predict(X_test)
    kwh_r2 = r2_score(yk_test, kwh_pred)

    score_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42
    )
    score_model.fit(X_train, ys_train)
    score_pred = score_model.predict(X_test)
    score_r2 = r2_score(ys_test, score_pred)

    print(f"R2 for monthly_kwh model: {kwh_r2:.3f}")
    print(f"R2 for sustainability_score model: {score_r2:.3f}")

    joblib.dump(
        {
            "kwh_model": kwh_model,
            "score_model": score_model,
            "feature_columns": X.columns.tolist(),
        },
        "lab_model.pkl",
    )
    print("Saved lab_model.pkl and synthetic_labs.csv")


if __name__ == "__main__":
    train_models()
