def read_data(
    filepath: str, separator: str = ' '
) -> tuple[list[tuple[int, int, int]], list[list[float]], dict]:
    from math import dist

    import pandas as pd

    data_csv = list(
        pd.read_csv(
            filepath, sep=separator, skipinitialspace=True, header=None
        ).to_records(index=False)
    )
    data_csv = [(int(row[0]), int(row[1]), int(row[2])) for row in data_csv]

    data_length = len(data_csv)
    distances_matrix = [[0.0] * data_length for _ in range(data_length)]
    for i in range(data_length):
        for j in range(data_length):
            if i == j:
                distances_matrix[i][j] = 0.0
            else:
                p1 = (data_csv[i][1], data_csv[i][2])
                p2 = (data_csv[j][1], data_csv[j][2])
                distances_matrix[i][j] = dist(p1, p2)

    index_to_id_map = {i: rec[0] for i, rec in enumerate(data_csv)}

    return data_csv, distances_matrix, index_to_id_map
