"""Simulated psychological research dataset for testing the application.

ALL VALUES ARE SIMULATED. They do not come from real respondents.
The generator uses a fixed random seed so everyone gets the same dataset.
A few data-quality problems are inserted on purpose (missing values, one
out-of-range Likert response, one duplicate row) so the cleaning tools
have something to find.
"""
import numpy as np
import pandas as pd

SAMPLE_DICTIONARY = pd.DataFrame(
    [
        ("Respondent_ID", "Identifier", "Unique respondent code (R001...)"),
        ("Age", "Numeric", "Age in years (18-35)"),
        ("Gender", "Categorical", "Female / Male / Non-binary"),
        ("Year_Level", "Categorical", "Academic year (1st-4th year)"),
        ("Condition", "Categorical", "Control or Mindfulness (simulated program)"),
        ("Sleep_Hours", "Numeric", "Average nightly sleep duration (hours)"),
        ("Stress_Score", "Numeric", "Baseline stress score (0-100, higher = more stress)"),
        ("Stress_Followup", "Numeric", "Stress score after 8 weeks (paired with Stress_Score)"),
        ("Motivation_Score", "Numeric", "Academic motivation (0-100)"),
        ("Academic_Productivity", "Numeric", "Academic productivity score (0-100)"),
        ("Q1-Q8", "Likert 1-5", "Perceived-stress questionnaire items "
                                "(1 = never ... 5 = very often); form one scale"),
    ],
    columns=["Variable", "Type", "Description"],
)


def generate_sample_data(n: int = 100, seed: int = 42,
                         include_quality_issues: bool = True) -> pd.DataFrame:
    """Build the simulated dataset (n respondents, plus 1 duplicate row)."""
    rng = np.random.default_rng(seed)

    ids = [f"R{i:03d}" for i in range(1, n + 1)]
    age = np.clip(np.round(rng.normal(21.5, 2.6, n)), 18, 35).astype(int)
    gender = rng.choice(["Female", "Male", "Non-binary"], n, p=[0.55, 0.40, 0.05])
    year = rng.choice(["1st year", "2nd year", "3rd year", "4th year"], n)
    condition = rng.permutation(np.array(["Control", "Mindfulness"] * (n // 2 + 1))[:n])

    sleep = np.clip(rng.normal(6.8, 1.1, n), 3.5, 9.5).round(1)
    z = rng.normal(0, 1, n)  # latent stress tendency
    stress = np.clip(52 + 9 * z - 3.0 * (sleep - 6.8) + rng.normal(0, 4, n), 10, 95)
    effect = np.where(condition == "Mindfulness", 6.0, 1.5)
    followup = np.clip(stress - effect + rng.normal(0, 6, n), 5, 100)
    motivation = np.clip(60 - 0.25 * (stress - 52) + rng.normal(0, 9, n), 10, 100)
    year_bonus = pd.Series(year).map(
        {"1st year": 0, "2nd year": 1, "3rd year": 2, "4th year": 3}).to_numpy()
    productivity = np.clip(
        55 + 0.35 * (motivation - 60) - 0.2 * (stress - 52)
        + 2.5 * (sleep - 6.8) + year_bonus + rng.normal(0, 7, n), 0, 100)

    zs = (stress - stress.mean()) / stress.std()
    items = {}
    for i in range(1, 9):
        raw = 3 + 0.9 * zs + rng.normal(0, 0.8, n)
        items[f"Q{i}"] = np.clip(np.round(raw), 1, 5).astype(int)

    df = pd.DataFrame({
        "Respondent_ID": ids, "Age": age, "Gender": gender, "Year_Level": year,
        "Condition": condition, "Sleep_Hours": sleep,
        "Stress_Score": stress.round(1), "Stress_Followup": followup.round(1),
        "Motivation_Score": motivation.round(1),
        "Academic_Productivity": productivity.round(1), **items,
    })

    if include_quality_issues:
        df.loc[7, "Sleep_Hours"] = np.nan
        df.loc[41, "Sleep_Hours"] = np.nan
        df.loc[63, "Q6"] = np.nan
        df["Q3"] = df["Q3"].astype(float)
        df.loc[29, "Q3"] = 7.0           # out of the 1-5 range
        df = pd.concat([df, df.iloc[[9]]], ignore_index=True)  # duplicate row
    return df


if __name__ == "__main__":
    from config import SAMPLE_FILE
    generate_sample_data().to_csv(SAMPLE_FILE, index=False)
    print(f"Saved {SAMPLE_FILE}")
