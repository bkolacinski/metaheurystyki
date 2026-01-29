import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.aco import aco_vrptw
from src.data_loader import read_to_solomon_data
from src.inter import relocate_operator
from src.solver import calculate_total_distance, three_opt_tw
from src.utils import calculate_distance_matrix


def run_multiple_times(
    instance_path: str,
    n_runs: int,
    n_ants: int,
    n_iterations: int,
    alpha: float,
    beta: float,
    rho: float,
    q0: float,
    use_local_search: bool = True,
    verbose: bool = True,
) -> Tuple:
    data = read_to_solomon_data(instance_path)
    dist_matrix = calculate_distance_matrix(data["coords"])

    if verbose:
        print(f"\n{'='*70}")
        print(f"Instancja: {data['name']}")
        print(
            f"Klienci: {len(data['coords']) - 1} | Pojemność: {data['capacity']} | Dostępne pojazdy: {data['n_vehicles']}"
        )
        print(
            f"Parametry: n_ants={n_ants}, n_iter={n_iterations}, α={alpha}, β={beta}, ρ={rho}, q0={q0}"
        )
        print(
            f"Optymalizacja lokalna: {'TAK (inter + intra)' if use_local_search else 'NIE'}"
        )
        print(f"\nUruchamianie {n_runs} razy...")
        print("=" * 70)

    best_routes = None
    best_n_vehicles = float("inf")
    best_distance = float("inf")
    all_results = []

    start_time = time.time()

    for run in range(n_runs):
        run_start = time.time()

        try:
            if verbose:
                print(f"\n[Run {run+1}/{n_runs}]", end=" ", flush=True)

            routes, n_veh, dist = aco_vrptw(
                distance_matrix=dist_matrix,
                service_times=data["service_times"],
                window_starts=data["window_starts"],
                window_ends=data["window_ends"],
                demands=data["demands"],
                capacity=float(data["capacity"]),
                n_vehicles_available=data["n_vehicles"],
                n_ants=n_ants,
                n_iterations=n_iterations,
                alpha=alpha,
                beta=beta,
                rho=rho,
                q0=q0,
                verbose=False,
                speed_factor=1.0,
            )

            aco_vehicles = n_veh
            aco_distance = dist
            if verbose:
                print(
                    f"ACO: {aco_vehicles}p/{aco_distance:.2f}d",
                    end=" ",
                    flush=True,
                )

            if use_local_search and routes:
                routes_list = [
                    route.tolist() if hasattr(route, "tolist") else route
                    for route in routes
                ]
                routes_np = [
                    np.array([0] + r + [0], dtype=np.int64)
                    for r in routes_list
                ]

                import contextlib
                import io

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    improved_routes = relocate_operator(
                        routes_np,
                        dist_matrix,
                        data["service_times"],
                        data["window_starts"],
                        data["window_ends"],
                        data["demands"],
                        float(data["capacity"]),
                        max_iterations=10,
                    )

                # Calculate after inter-route
                temp_dist = sum(
                    calculate_total_distance(r, dist_matrix)
                    for r in improved_routes
                )
                inter_vehicles = len(
                    [r for r in improved_routes if len(r) > 2]
                )
                if verbose:
                    print(
                        f"→ Inter: {inter_vehicles}p/{temp_dist:.2f}d",
                        end=" ",
                        flush=True,
                    )

                optimized = []
                for route in improved_routes:
                    if len(route) <= 2:
                        continue

                    optimized_route = three_opt_tw(
                        route.tolist(),
                        dist_matrix,
                        data["service_times"],
                        data["window_starts"],
                        data["window_ends"],
                        speed_factor=1.0,
                    )
                    route_without_depot = optimized_route[1:-1].tolist()
                    if len(route_without_depot) > 0:
                        optimized.append(route_without_depot)

                routes = optimized
                n_veh = len(routes)

                dist = sum(
                    calculate_total_distance(
                        np.array([0] + r + [0], dtype=np.int64), dist_matrix
                    )
                    for r in routes
                )

                if verbose:
                    print(
                        f"→ Intra: {n_veh}p/{dist:.2f}d", end=" ", flush=True
                    )

            run_time = time.time() - run_start

            all_results.append(
                {
                    "run": run + 1,
                    "n_vehicles": n_veh,
                    "distance": dist,
                    "time": run_time,
                    "routes": routes,
                }
            )

            is_better = False
            if n_veh < best_n_vehicles:
                is_better = True
            elif n_veh == best_n_vehicles and dist < best_distance:
                is_better = True

            if is_better:
                best_routes = routes
                best_n_vehicles = n_veh
                best_distance = dist
                status = "✓ NAJLEPSZE"
            else:
                status = ""

            if verbose:
                print(f"[{run_time:.1f}s] {status}")

        except Exception as e:
            if verbose:
                print(f"BŁĄD: {e}")
            all_results.append(
                {
                    "run": run + 1,
                    "n_vehicles": float("inf"),
                    "distance": float("inf"),
                    "time": 0,
                    "routes": None,
                }
            )

    total_time = time.time() - start_time

    valid_results = [r for r in all_results if r["n_vehicles"] != float("inf")]

    worst_distance = float("-inf")
    worst_routes = None
    worst_n_vehicles = 0
    avg_routes = None
    avg_distance = 0.0
    avg_n_vehicles = 0
    std_distance = 0.0

    if valid_results:
        distances = [r["distance"] for r in valid_results]
        vehicles = [r["n_vehicles"] for r in valid_results]

        for result in valid_results:
            if result["distance"] > worst_distance:
                worst_distance = result["distance"]
                worst_routes = result.get("routes")
                worst_n_vehicles = result["n_vehicles"]

        avg_distance = sum(distances) / len(distances)
        avg_n_vehicles = sum(vehicles) / len(vehicles)
        std_distance = np.std(distances)

        closest_to_avg = min(
            valid_results, key=lambda r: abs(r["distance"] - avg_distance)
        )
        avg_routes = closest_to_avg.get("routes")

    if verbose:
        print("\n" + "=" * 70)
        print(f"Zakończono {n_runs} uruchomień w {total_time:.2f}s")
        print("\n✓ NAJLEPSZY WYNIK:")
        print(f"  Pojazdy: {best_n_vehicles}")
        print(f"  Dystans: {best_distance:.2f}")
        print(f"  Liczba tras: {len(best_routes) if best_routes else 0}")

        if valid_results:
            print("\n✗ NAJGORSZY WYNIK:")
            print(f"  Pojazdy: {worst_n_vehicles}")
            print(f"  Dystans: {worst_distance:.2f}")

            print("\n≈ ŚREDNI WYNIK:")
            print(f"  Pojazdy: {avg_n_vehicles:.2f}")
            print(f"  Dystans: {avg_distance:.2f}")

            print("\nODCHYLENIE STANDARDOWE:")
            print(f"  Dystans: {std_distance:.2f}")

            print("\nCZAS OBLICZEŃ:")
            print(f"  Całkowity: {total_time:.2f}s")
            print(f"  Średni na uruchomienie: {total_time/n_runs:.2f}s")

            vehicles_list = [r["n_vehicles"] for r in valid_results]
            distances_list = [r["distance"] for r in valid_results]
            print(f"\nStatystyki ({len(valid_results)} udanych uruchomień):")
            print(
                f"  Pojazdy - Min: {min(vehicles_list)}, Max: {max(vehicles_list)}, "
                f"Średnia: {avg_n_vehicles:.2f}"
            )
            print(
                f"  Dystans - Min: {min(distances_list):.2f}, Max: {max(distances_list):.2f}, "
                f"Średnia: {avg_distance:.2f}, Std: {std_distance:.2f}"
            )

    return (
        best_routes,
        best_n_vehicles,
        best_distance,
        worst_routes,
        worst_n_vehicles,
        worst_distance,
        avg_routes,
        avg_n_vehicles,
        avg_distance,
        std_distance,
        total_time,
        all_results,
        data,
        dist_matrix,
    )


def visualize_solution(
    routes: List,
    coords: np.ndarray,
    instance_name: str,
    n_vehicles: int,
    distance: float,
    params: Dict,
    output_dir: str = "results",
    suffix: str = "best_solution",
    title_prefix: str = "Najlepsze rozwiązanie",
):
    sns.set_style("whitegrid")

    _, ax = plt.subplots(figsize=(14, 10))

    colors = sns.color_palette("husl", len(routes))

    depot_x, depot_y = coords[0]
    ax.scatter(
        depot_x,
        depot_y,
        c="red",
        s=300,
        marker="s",
        zorder=5,
        edgecolors="black",
        linewidths=2,
        label="Magazyn",
    )
    ax.text(
        depot_x,
        depot_y,
        "M",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
    )

    for route_idx, route in enumerate(routes):
        color = colors[route_idx]

        if hasattr(route, "tolist"):
            route_list = route.tolist()
        else:
            route_list = route

        if route_list and route_list[0] == 0:
            route_list = route_list[1:]
        if route_list and route_list[-1] == 0:
            route_list = route_list[:-1]

        full_route = [0] + route_list + [0]
        route_coords = coords[full_route]

        ax.plot(
            route_coords[:, 0],
            route_coords[:, 1],
            c=color,
            linewidth=2,
            alpha=0.7,
            zorder=2,
            label=f"Trasa {route_idx+1} ({len(route_list)} klientów)",
        )

        for customer_idx in route_list:
            cx, cy = coords[customer_idx]
            ax.scatter(
                cx,
                cy,
                c=[color],
                s=100,
                zorder=3,
                edgecolors="black",
                linewidths=1,
            )

    title = f"{title_prefix}: {instance_name}\n"
    title += f"Pojazdy: {n_vehicles} | Dystans: {distance:.2f}\n"
    title += f"α={params['alpha']}, β={params['beta']}, ρ={params['rho']}, "
    title += f"q0={params['q0']}, mrówki={params['n_ants']}, iteracje={params['n_iterations']}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Współrzędna X", fontsize=12)
    ax.set_ylabel("Współrzędna Y", fontsize=12)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_base = os.path.basename(instance_name).replace(".txt", "")
    filename = f"{instance_base}_{suffix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Wykres zapisany: {filepath}")

    plt.close()


def save_statistics(
    all_results: List[Dict],
    instance_name: str,
    params: Dict,
    best_distance: float,
    worst_distance: float,
    avg_distance: float,
    std_distance: float,
    total_time: float,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_base = os.path.basename(instance_name).replace(".txt", "")

    csv_filename = f"{instance_base}_runs_{timestamp}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)

    with open(csv_filepath, "w") as f:
        f.write("run,n_vehicles,distance,time\n")

        for result in all_results:
            f.write(
                f"{result['run']},{result['n_vehicles']},"
                f"{result['distance']:.2f},{result['time']:.2f}\n"
            )

    print(f"Statystyki zapisane: {csv_filepath}")

    summary_filename = f"{instance_base}_summary_{timestamp}.txt"
    summary_filepath = os.path.join(output_dir, summary_filename)

    valid_results = [r for r in all_results if r["n_vehicles"] != float("inf")]

    with open(summary_filepath, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"PODSUMOWANIE WYNIKÓW: {instance_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write("PARAMETRY:\n")
        f.write(f"  n_ants: {params['n_ants']}\n")
        f.write(f"  n_iterations: {params['n_iterations']}\n")
        f.write(f"  alpha: {params['alpha']}\n")
        f.write(f"  beta: {params['beta']}\n")
        f.write(f"  rho: {params['rho']}\n")
        f.write(f"  q0: {params['q0']}\n\n")

        f.write("WYNIKI:\n")
        f.write(f"  Najlepszy wynik: {best_distance:.2f}\n")
        f.write(f"  Najgorszy wynik: {worst_distance:.2f}\n")
        f.write(f"  Średni wynik: {avg_distance:.2f}\n")
        f.write(f"  Odchylenie standardowe: {std_distance:.2f}\n")
        f.write(f"  Czas obliczeń: {total_time:.2f}s\n\n")

        f.write(f"Liczba uruchomień: {len(all_results)}\n")
        f.write(f"Udane uruchomienia: {len(valid_results)}\n")
        f.write("=" * 70 + "\n")

    print(f"Podsumowanie zapisane: {summary_filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ACO VRPTW multiple times and visualize best solution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "instance",
        help="Instance file name (e.g., c107.txt)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs (default: 10)",
    )
    parser.add_argument(
        "--n_ants",
        type=int,
        default=200,
        help="Number of ants per iteration (default: 200)",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=100,
        help="Number of iterations (default: 100)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Pheromone importance (default: 1.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=3.0,
        help="Heuristic importance (default: 3.0)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.1,
        help="Evaporation rate (default: 0.1)",
    )
    parser.add_argument(
        "--q0",
        type=float,
        default=0.9,
        help="Exploitation threshold (default: 0.9)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for plots and statistics (default: results/)",
    )
    parser.add_argument(
        "--no-local-search",
        action="store_true",
        help="Disable local search optimization",
    )

    args = parser.parse_args()

    instance_path = os.path.join("data", args.instance)

    if not os.path.exists(instance_path):
        print(f"ERROR: Instance file not found: {instance_path}")
        sys.exit(1)

    params = {
        "n_ants": args.n_ants,
        "n_iterations": args.n_iterations,
        "alpha": args.alpha,
        "beta": args.beta,
        "rho": args.rho,
        "q0": args.q0,
    }

    (
        best_routes,
        best_n_vehicles,
        best_distance,
        worst_routes,
        worst_n_vehicles,
        worst_distance,
        avg_routes,
        avg_n_vehicles,
        avg_distance,
        std_distance,
        total_time,
        all_results,
        data,
        _,
    ) = run_multiple_times(
        instance_path=instance_path,
        n_runs=args.runs,
        n_ants=args.n_ants,
        n_iterations=args.n_iterations,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        q0=args.q0,
        use_local_search=not args.no_local_search,
        verbose=True,
    )

    if best_routes is None:
        print("\n❌ BŁĄD: Nie znaleziono poprawnego rozwiązania!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("TWORZENIE WIZUALIZACJI")
    print("=" * 70)
    visualize_solution(
        routes=best_routes,
        coords=data["coords"],
        instance_name=args.instance,
        n_vehicles=best_n_vehicles,
        distance=best_distance,
        params=params,
        output_dir=args.output,
        suffix="best_solution",
        title_prefix="Najlepsze rozwiązanie",
    )

    if worst_routes:
        visualize_solution(
            routes=worst_routes,
            coords=data["coords"],
            instance_name=args.instance,
            n_vehicles=worst_n_vehicles,
            distance=worst_distance,
            params=params,
            output_dir=args.output,
            suffix="worst_solution",
            title_prefix="Najgorsze rozwiązanie",
        )

    if avg_routes:
        visualize_solution(
            routes=avg_routes,
            coords=data["coords"],
            instance_name=args.instance,
            n_vehicles=int(avg_n_vehicles),
            distance=avg_distance,
            params=params,
            output_dir=args.output,
            suffix="avg_solution",
            title_prefix="Średnie rozwiązanie",
        )

    save_statistics(
        all_results=all_results,
        instance_name=args.instance,
        params=params,
        best_distance=best_distance,
        worst_distance=worst_distance,
        avg_distance=avg_distance,
        std_distance=std_distance,
        total_time=total_time,
        output_dir=args.output,
    )

    print("\n" + "=" * 70)
    print("ZAKOŃCZONO POMYŚLNIE")
    print("=" * 70)


if __name__ == "__main__":
    main()
