def read_data(filepath: str) -> tuple[
    list[tuple[int, int, int]],
    list[list[float]]
]:
    import pandas as pd
    from math import dist
    data_csv = list(pd.read_csv(filepath, sep=' ', skipinitialspace=True,
                                header=None).to_records(index=False))

    distances_matrix = []
    for i in range(len(data_csv)):
        distances_matrix.append([])
        for j in range(len(data_csv)):
            if i == j:
                distances_matrix[i].append(0)
            else:
                distances_matrix[i].append(dist(data_csv[i], data_csv[j]))

    return data_csv, distances_matrix
