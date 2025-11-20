def read_data_csv(file_path: str) -> list[dict]:
    import pandas as pd

    data_csv = pd.read_csv(
        filepath_or_buffer=file_path,
        delimiter="\t", index_col=0
    )
    data_csv.rename_axis(None)
    data_csv.columns = ["Nazwa", "Waga", "Wartosc"]
    data_csv["Waga"] = (data_csv["Waga"]
                        .astype(str)
                        .str.replace(" ", "")
                        .astype(int))
    data_csv["Wartosc"] = (
        data_csv["Wartosc"].astype(str).str.replace(" ", "").astype(int)
    )

    return data_csv