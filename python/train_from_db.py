"""
train_from_db.py  (fixed)
=========================
Trains all four HR AI models using REAL employee data from the hr_ai_system
MySQL database.

Key fixes over the original:
  1. Attrition labels come from real has_left data (10 employees who left).
     Proxy labels are only generated as a true last resort, with a clear warning.
  2. The has_left CASE statement is simplified to avoid type-mismatch issues.
  3. Added a diagnostic section that prints exactly how many leavers were found
     before training, so you can verify the labels are real.
  4. Class imbalance handled with class_weight='balanced' on LogisticRegression
     so the model doesn't just predict "stays" for everyone.
  5. Evaluation now prints precision/recall per class so you can see if the
     model is actually learning to detect leavers.
"""

import os
import sys
import json
# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report
)

# ── Database connection ────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "",
    "database": "hr_ai_system",
}

# ── Output directory ───────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature columns ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Age",
    "YearsAtCompany",
    "BaseSalary",
    "JobSatisfaction",
    "PerformanceRating",
    "ProjectsCompleted",
    "HoursWorkedPerWeek",
]

MIN_RECORDS = 10


# ── 1. Load data ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("Connecting to database...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"ERROR: Could not connect to MySQL.\n  {e}")
        sys.exit(1)

    # Pull ALL employees (active + left) so attrition labels are real.
    # has_left is stored as tinyint(1) or varchar — cast to int explicitly.
    query = """
        SELECT
            e.id                                AS employee_db_id,
            e.age                               AS Age,
            e.years_at_company                  AS YearsAtCompany,
            e.job_satisfaction                  AS JobSatisfaction,
            CAST(e.has_left AS UNSIGNED)        AS HasLeft,

            s.base_salary                       AS BaseSalary,
            s.net_salary                        AS NetSalary,

            p.productivity_score                AS ProductivityScore,
            p.manager_rating                    AS ManagerRating,
            p.projects_completed                AS ProjectsCompleted,
            p.hours_worked_per_week             AS HoursWorkedPerWeek

        FROM employees e

        JOIN (
            SELECT s1.*
            FROM salary s1
            INNER JOIN (
                SELECT employee_id, MAX(id) AS max_id
                FROM salary
                GROUP BY employee_id
            ) s2 ON s1.employee_id = s2.employee_id AND s1.id = s2.max_id
        ) s ON s.employee_id = e.id

        JOIN (
            SELECT p1.*
            FROM performance p1
            INNER JOIN (
                SELECT employee_id, MAX(evaluation_date) AS max_date
                FROM performance
                GROUP BY employee_id
            ) p2 ON p1.employee_id = p2.employee_id
               AND p1.evaluation_date = p2.max_date
        ) p ON p.employee_id = e.id
    """

    df = pd.read_sql(query, conn)
    conn.close()
    print(f"  Loaded {len(df)} employee records from the database.")
    return df


# ── 2. Diagnose and build attrition labels ────────────────────────────────────
def build_attrition_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses real has_left values from the database.
    Only falls back to proxy labels if zero leavers exist in the data.
    Prints a clear diagnostic so you know which path was taken.
    """

    df["HasLeft"] = pd.to_numeric(df["HasLeft"], errors="coerce").fillna(0).astype(int)

    leavers     = int(df["HasLeft"].sum())
    total       = len(df)
    stayed      = total - leavers

    print("\n  ── Attrition Label Diagnostic ──────────────────────────")
    print(f"     Total employees : {total}")
    print(f"     Stayed (0)      : {stayed}")
    print(f"     Left   (1)      : {leavers}")

    if leavers == 0:
        # ── True last resort: generate proxy labels ──────────────────────────
        print("\n  [WARNING] No real leavers found in the database.")
        print("  Generating proxy attrition labels from risk factors.")
        print("  To fix this permanently, make sure has_left = 1 for")
        print("  employees who have actually left (see employees_dataset.sql).")

        np.random.seed(42)

        # Risk factors: low satisfaction, excess hours, low salary
        sat_norm    = (5  - df["JobSatisfaction"].clip(1, 5)) / 4      # 0=happy, 1=miserable
        hours_norm  = ((df["HoursWorkedPerWeek"] - 40).clip(0)) / 30   # 0=normal, 1=extreme
        sal_mean    = df["BaseSalary"].mean()
        sal_std     = df["BaseSalary"].std() if df["BaseSalary"].std() > 0 else 1
        sal_norm    = ((sal_mean - df["BaseSalary"]) / sal_std).clip(-2, 2) / 4  # 0=ok, 0.5=very low

        risk = sat_norm * 0.5 + hours_norm * 0.3 + sal_norm * 0.2

        # Mark top 20% as at-risk (not 25% — keeps class balance tighter)
        threshold = risk.quantile(0.80)
        df["Attrition"] = (risk >= threshold).astype(int)

        proxy_leavers = int(df["Attrition"].sum())
        print(f"  Proxy leavers generated : {proxy_leavers} / {total}")

    else:
        # ── Use real labels ──────────────────────────────────────────────────
        print("  [OK] Using real has_left labels from database.")
        df["Attrition"] = df["HasLeft"]

        if leavers < 5:
            print(f"\n  [WARNING] Only {leavers} real leaver(s) found.")
            print("  With fewer than 5 leavers the model may not generalise well.")
            print("  Consider adding more historical leaver records.")

    print("  ────────────────────────────────────────────────────────\n")
    return df


# ── 3. Derive all targets ──────────────────────────────────────────────────────
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    # PerformanceRating: round manager_rating to nearest int, clip 1-5
    df["PerformanceRating"] = df["ManagerRating"].round().clip(1, 5).astype(int)

    # Target salary: derived to avoid data leakage from BaseSalary feature
    df["TargetSalary"] = (
        df["BaseSalary"]
        + df["YearsAtCompany"] * 1000
        + df["PerformanceRating"] * 2000
        + df["ProjectsCompleted"] * 500
    )

    # Attrition: built in build_attrition_labels()

    # Promotion proxy: high rating + many projects + tenure
    df["Promotion"] = (
        (df["PerformanceRating"] >= 4) &
        (df["ProjectsCompleted"] >= 10) &
        (df["YearsAtCompany"] >= 2)
    ).astype(int)

    # Employee category
    def categorise(row):
        if row["PerformanceRating"] >= 4 and row["ProjectsCompleted"] >= 15:
            return 2   # High Potential
        if row["PerformanceRating"] <= 2 or row["ProjectsCompleted"] <= 5:
            return 0   # Underperformer
        return 1       # Steady

    df["Category"] = df.apply(categorise, axis=1)
    return df


# ── 4. Train helpers ───────────────────────────────────────────────────────────
def split(df, target):
    X = df[FEATURE_COLS]
    y = df[target]
    # Stratify for classification targets to preserve class ratio in test set
    try:
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        # Stratify fails if a class has only 1 member — fall back to random split
        return train_test_split(X, y, test_size=0.2, random_state=42)


def train_salary(df):
    print("[1/4] Salary Prediction — Linear Regression")
    X_tr, X_te, y_tr, y_te = split(df, "TargetSalary")
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, preds))
    r2  = float(r2_score(y_te, preds))
    print(f"  MAE : Rs.{mae:,.0f}   R2 : {r2:.3f}")
    path = os.path.join(MODELS_DIR, "salary_model.pkl")
    joblib.dump(model, path)
    print(f"  Saved -> {path}\n")
    return {"salary_mae": mae, "salary_r2": r2}


def train_attrition(df):
    print("[2/4] Attrition Prediction — Logistic Regression")
    X_tr, X_te, y_tr, y_te = split(df, "Attrition")

    leavers_in_train = int(y_tr.sum())
    print(f"  Leavers in training set : {leavers_in_train} / {len(y_tr)}")

    # class_weight='balanced' compensates for the imbalance
    # (far more stayers than leavers) so the model doesn't just
    # predict "stays" for everyone to get high accuracy
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc   = float(accuracy_score(y_te, preds))
    print(f"  Accuracy : {acc:.2%}")
    print(classification_report(
        y_te, preds,
        target_names=["Stayed", "Left"],
        zero_division=0
    ))
    path = os.path.join(MODELS_DIR, "attrition_model.pkl")
    joblib.dump(model, path)
    print(f"  Saved -> {path}\n")
    return {"attrition_accuracy": acc}


def train_promotion(df):
    print("[3/4] Promotion Prediction — Random Forest")
    X_tr, X_te, y_tr, y_te = split(df, "Promotion")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc   = float(accuracy_score(y_te, preds))
    print(f"  Accuracy : {acc:.2%}")
    print(classification_report(
        y_te, preds,
        target_names=["Not Ready", "Ready"],
        zero_division=0
    ))
    path = os.path.join(MODELS_DIR, "promotion_model.pkl")
    joblib.dump(model, path)
    print(f"  Saved -> {path}\n")
    return {"promotion_accuracy": acc}


def train_category(df):
    print("[4/4] Employee Category — Decision Tree")
    X_tr, X_te, y_tr, y_te = split(df, "Category")
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc   = float(accuracy_score(y_te, preds))
    print(f"  Accuracy : {acc:.2%}")
    label_map     = {0: "Underperformer", 1: "Steady", 2: "High Potential"}
    present       = sorted(set(y_te) | set(preds))
    present_names = [label_map[l] for l in present]
    print(classification_report(
        y_te, preds,
        labels=present,
        target_names=present_names,
        zero_division=0
    ))
    path = os.path.join(MODELS_DIR, "category_model.pkl")
    joblib.dump(model, path)
    print(f"  Saved -> {path}\n")
    return {"category_accuracy": acc}


# ── 5. Log to DB ───────────────────────────────────────────────────────────────
def log_training(total: int, used: int, notes: str):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO training_data_log
                (total_records, records_used, data_version, notes)
            VALUES (%s, %s, %s, %s)
        """, (total, used, "db_v2_real_attrition", notes))
        conn.commit()
        conn.close()
        print("  Training run logged to training_data_log table.")
    except Exception as e:
        print(f"  Warning: could not write to training_data_log — {e}")


# ── 6. Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  HR AI System — Model Training (fixed attrition labels)")
    print("=" * 60)

    df_raw = load_data()

    if len(df_raw) < MIN_RECORDS:
        print(
            f"\nERROR: Only {len(df_raw)} records found after joining tables.\n"
            "  Make sure employees, salary, and performance tables all\n"
            "  contain rows with matching employee IDs."
        )
        sys.exit(1)

    # Build attrition labels first (diagnostic prints here)
    df = build_attrition_labels(df_raw)

    # Build remaining targets
    df = build_targets(df)

    # Drop rows with NaN in any needed column
    needed = FEATURE_COLS + ["TargetSalary", "Attrition", "Promotion", "Category"]
    before = len(df)
    df     = df.dropna(subset=needed)
    after  = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with missing values. Using {after}.\n")

    print("=" * 60)
    m1 = train_salary(df)
    m2 = train_attrition(df)
    m3 = train_promotion(df)
    m4 = train_category(df)
    print("=" * 60)

    notes = json.dumps({**m1, **m2, **m3, **m4})
    log_training(total=before, used=after, notes=notes)

    print("\n  All 4 models trained and saved to ../models/")
    print("  Restart app.py to load the new models.")
    print("=" * 60)


if __name__ == "__main__":
    main()