from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "Dane" / "Przetworzone" / "sales_all_months.csv"
OUT_PATH = ROOT / "Dane" / "Czyste" / "sales_clean.csv"


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    sales_clean = sales.copy()

    distance_cols = [
        "collegeDistance",
        "clinicDistance",
        "restaurantDistance",
        "pharmacyDistance",
        "postOfficeDistance",
        "kindergartenDistance",
        "schoolDistance",
    ]

    # 1. Drop condition because it is missing in a very large fraction of rows.
    if "condition" in sales_clean.columns:
        sales_clean = sales_clean.drop(columns=["condition"])

    # 2. Impute buildingMaterial only for blockOfFlats, using a hot-deck/random strategy.
    if "buildingMaterial" in sales_clean.columns and "type" in sales_clean.columns:
        block_mask = sales_clean["type"].eq("blockOfFlats")
        missing_mask = block_mask & sales_clean["buildingMaterial"].isna()

        if missing_mask.any():
            observed = sales_clean.loc[
                block_mask & sales_clean["buildingMaterial"].notna(),
                ["city", "ownership", "buildingMaterial"],
            ]
            if not observed.empty:
                rng = np.random.default_rng(42)
                filled = []
                for _, row in sales_clean.loc[missing_mask, ["city", "ownership"]].iterrows():
                    city = row["city"]
                    ownership = row["ownership"]
                    candidates = observed.loc[
                        (
                            observed["city"].fillna("Unknown")
                            == (city if pd.notna(city) else "Unknown")
                        )
                        & (
                            observed["ownership"].fillna("Unknown")
                            == (ownership if pd.notna(ownership) else "Unknown")
                        ),
                        "buildingMaterial",
                    ]
                    if candidates.empty:
                        candidates = observed.loc[
                            observed["city"].fillna("Unknown")
                            == (city if pd.notna(city) else "Unknown"),
                            "buildingMaterial",
                        ]
                    if candidates.empty:
                        candidates = observed["buildingMaterial"]
                    filled.append(candidates.sample(1, random_state=int(rng.integers(0, 100000))).iloc[0])
                sales_clean.loc[missing_mask, "buildingMaterial"] = filled

        sales_clean["buildingMaterial"] = sales_clean["buildingMaterial"].fillna("Unknown")

    # 3. Fill the remaining categorical columns conservatively.
    categorical_cols = [
        "city",
        "type",
        "ownership",
        "hasParkingSpace",
        "hasBalcony",
        "hasElevator",
        "hasSecurity",
        "hasStorageRoom",
        "month",
    ]
    for col in categorical_cols:
        if col in sales_clean.columns:
            sales_clean[col] = sales_clean[col].fillna("Unknown")

    # 4. Numeric columns: use group-wise median by type where possible.
    numeric_cols = ["buildYear", "floor", "floorCount", "squareMeters", "rooms", "poiCount"]
    for col in numeric_cols:
        if col in sales_clean.columns:
            if "type" in sales_clean.columns:
                sales_clean[col] = sales_clean.groupby("type")[col].transform(
                    lambda s: s.fillna(s.median())
                )
            sales_clean[col] = sales_clean[col].fillna(sales_clean[col].median())

    # 5. Distance columns: use a MICE-like multivariate imputation with auxiliary variables.
    if all(col in sales_clean.columns for col in distance_cols):
        feature_cols = [
            "squareMeters",
            "rooms",
            "floor",
            "poiCount",
            "buildYear",
            "type",
            "city",
            "ownership",
            "buildingMaterial",
            "hasElevator",
            "hasParkingSpace",
            "hasBalcony",
            "hasSecurity",
            "hasStorageRoom",
        ]
        impute_frame = sales_clean[distance_cols + [col for col in feature_cols if col in sales_clean.columns]].copy()

        cat_cols = [col for col in ["type", "city", "ownership", "buildingMaterial"] if col in impute_frame.columns]
        for col in cat_cols:
            impute_frame[col] = impute_frame[col].fillna("Unknown")

        impute_frame = pd.get_dummies(impute_frame, columns=cat_cols, dummy_na=False)
        for col in impute_frame.columns:
            if impute_frame[col].isna().any():
                impute_frame[col] = impute_frame[col].fillna(impute_frame[col].median())

        imputer = IterativeImputer(random_state=0, max_iter=25, initial_strategy="median")
        imputed = pd.DataFrame(
            imputer.fit_transform(impute_frame),
            columns=impute_frame.columns,
            index=sales_clean.index,
        )
        sales_clean[distance_cols] = imputed[distance_cols]

    # 6. Fill any remaining missing values conservatively.
    for col in sales_clean.columns:
        if sales_clean[col].isna().any():
            if pd.api.types.is_numeric_dtype(sales_clean[col]):
                sales_clean[col] = sales_clean[col].fillna(sales_clean[col].median())
            else:
                sales_clean[col] = sales_clean[col].fillna("Unknown")

    return sales_clean


if __name__ == "__main__":
    sales = pd.read_csv(RAW_PATH)
    sales_clean = clean_sales(sales)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sales_clean.to_csv(OUT_PATH, index=False)
    print(f"Zapisano {len(sales_clean)} rekordów do {OUT_PATH}")
    print("Pozostałe braki:")
    print(sales_clean.isna().sum().sort_values(ascending=False))
