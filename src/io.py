import pandas as pd

def extract_date_from_page(key: pd.DataFrame) -> pd.DataFrame:
    parts = key["page"].str.rsplit("_", n=1, expand=True)
    key["page"] = parts[0]
    key["date"] = pd.to_datetime(parts[1], errors="coerce")

    if key["date"].isna().any():
        raise ValueError("Failed to parse date from key page column.")

    return key

def read_key(path: str) -> pd.DataFrame:
    key = pd.read_csv(path, compression="zip")
    key = key.rename(columns={c: c.lower() for c in key.columns})

    if "id" not in key.columns or "page" not in key.columns:
        raise ValueError(f"Key missing required columns. Found: {list(key.columns)}")

    key = extract_date_from_page(key)
    return key[["id", "page", "date"]]
