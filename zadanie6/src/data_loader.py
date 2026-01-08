from dataclasses import dataclass
from typing import TypedDict

import numpy as np


@dataclass
class SolomonData(TypedDict):
    name: str
    n_vehicles: int
    capacity: int
    coords: np.ndarray
    demands: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    service_times: np.ndarray


def read_to_solomon_data(file_path: str) -> SolomonData:

    with open(file_path, "r") as file:
        lines: list[str] = file.readlines()

    name: str = lines[0].strip()

    n_vehicles: int = 0
    capacity: int = 0

    for i, line in enumerate(lines):
        if "NUMBER" in line and "CAPACITY" in line:
            parts: list[str] = lines[i + 1].split()
            n_vehicles = int(parts[0])
            capacity = int(parts[1])
            break

    customer_raw_data: list[list[float]] = []
    start_parsing: bool = False

    for line in lines:
        parts = line.split()
        if not parts:
            continue

        # Nagłówek danych klientów
        if parts[0] == "CUST" and parts[1] == "NO.":
            start_parsing = True
            continue

        if start_parsing:
            try:
                # Format rzędu: [ID, X, Y, DEMAND, READY, DUE, SERVICE]
                row: list[float] = [float(x) for x in parts]
                customer_raw_data.append(row)
            except ValueError:
                # Pomiń linie, które nie są danymi numerycznymi
                continue

    # Konwersja na macierz NumPy dla wydajności (Numba ją pokocha)
    data_matrix: np.ndarray = np.array(customer_raw_data, dtype=np.float64)

    result: SolomonData = {
        "name": name,
        "n_vehicles": n_vehicles,
        "capacity": capacity,
        "coords": data_matrix[:, 1:3],
        "demands": data_matrix[:, 3],
        "window_starts": data_matrix[:, 4],
        "window_ends": data_matrix[:, 5],
        "service_times": data_matrix[:, 6],
    }

    return result


if __name__ == "__main__":
    data = read_to_solomon_data("../data/c107.txt")
    print(data)
