import pandas as pd

def concat_and_dedupe(preds, keep="first", verbose=False) -> pd.DataFrame:
    out = pd.concat(preds, axis=0, ignore_index=True)

    if verbose:
        dup_ids = out.loc[out["Id"].duplicated(keep=False), "Id"].nunique()
        if dup_ids:
            print(f"[WARN] duplicated Id(s): {dup_ids}")

    out = out.drop_duplicates(subset="Id", keep=keep).reset_index(drop=True)
    assert not out["Id"].duplicated().any()
    return out
