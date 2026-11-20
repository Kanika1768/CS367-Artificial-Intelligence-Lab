import math
from pathlib import Path
import numpy as np
import scipy.io
from PIL import Image

INPUT_FILE = Path(__file__).resolve().parent / "scrambled_lena.mat"

TILE_SIZE = 128
GRID = 4
SEED = 7
RESTARTS = 30
STEPS = 60000
STABLE_WINDOW = 5000


def edge_costs(tiles):
    n = len(tiles)
    horizontal = np.zeros((n, n), dtype=float)
    vertical = np.zeros((n, n), dtype=float)

    for i, a in enumerate(tiles):
        for j, b in enumerate(tiles):
            horizontal[i, j] = np.mean(
                np.abs(
                    a[4:-4, -1].astype(float)
                    - b[4:-4, 0].astype(float)
                )
            )

            vertical[i, j] = np.mean(
                np.abs(
                    a[-1, 4:-4].astype(float)
                    - b[0, 4:-4].astype(float)
                )
            )

    return horizontal, vertical


def objective(order, horizontal, vertical):
    grid = order.reshape(GRID, GRID)

    return float(
        np.sum(
            horizontal[
                grid[:, :-1],
                grid[:, 1:]
            ]
        )
        + np.sum(
            vertical[
                grid[:-1, :],
                grid[1:, :]
            ]
        )
    )


def simulated_annealing(horizontal, vertical, seed=SEED):
    rng = np.random.default_rng(seed)

    best_order = None
    best_score = float("inf")
    global_best_restart = None
    global_best_iteration = None

    stability_results = []

    for restart in range(RESTARTS):
        print("\n" + "=" * 55)
        print(f"Restart {restart + 1}/{RESTARTS}")
        print("=" * 55)

        current = rng.permutation(GRID * GRID)

        current_score = objective(
            current, horizontal, vertical
        )

        local_best = current.copy()
        local_score = current_score

        last_improvement_iteration = 0
        first_stable_iteration = None

        for step in range(STEPS):
            iteration = step + 1

            i, j = rng.choice(
                GRID * GRID,
                size=2,
                replace=False
            )

            current[i], current[j] = (
                current[j], current[i]
            )

            candidate_score = objective(
                current, horizontal, vertical
            )

            progress = step / max(1, STEPS - 1)

            temperature = (
                20.0 * (1.0 - progress) + 0.05
            )

            accept = (
                candidate_score < current_score
                or rng.random() < math.exp(
                    (current_score - candidate_score)
                    / temperature
                )
            )

            if accept:
                current_score = candidate_score
            else:
                current[i], current[j] = (
                    current[j], current[i]
                )

            if current_score < local_score:
                local_score = current_score
                local_best = current.copy()

                last_improvement_iteration = iteration
                first_stable_iteration = None

            iterations_without_improvement = (
                iteration - last_improvement_iteration
            )

            if (
                first_stable_iteration is None
                and iterations_without_improvement
                >= STABLE_WINDOW
            ):
                first_stable_iteration = (
                    last_improvement_iteration
                    + STABLE_WINDOW
                )

                print(
                    f"\nStability criterion first met at "
                    f"iteration {first_stable_iteration}"
                )

                print(
                    f"No improvement in best cost for "
                    f"{STABLE_WINDOW} consecutive iterations."
                )

            if iteration % 10000 == 0:
                print(
                    f"Iteration: {iteration:6d} | "
                    f"Current cost: {current_score:.3f} | "
                    f"Best cost: {local_score:.3f} | "
                    f"Temperature: {temperature:.4f}"
                )

        if (
            STEPS - last_improvement_iteration
            >= STABLE_WINDOW
        ):
            final_stable_iteration = (
                last_improvement_iteration
                + STABLE_WINDOW
            )

            print(
                f"\nFinal stabilization iteration: "
                f"{final_stable_iteration}"
            )

            print(
                f"Best cost did not improve after that "
                f"iteration through iteration {STEPS}."
            )
        else:
            final_stable_iteration = None

            print(
                "\nStability criterion was not maintained "
                "for the required window at the end."
            )

        print(
            f"Final best cost for restart "
            f"{restart + 1}: {local_score:.3f}"
        )

        print(
            f"Last best-cost improvement: "
            f"iteration {last_improvement_iteration}"
        )

        stability_results.append({
            "restart": restart + 1,
            "best_cost": local_score,
            "last_improvement_iteration":
                last_improvement_iteration,
            "stable_iteration": final_stable_iteration
        })

        if local_score < best_score:
            best_score = local_score
            best_order = local_best.copy()

            global_best_restart = restart + 1
            global_best_iteration = (
                last_improvement_iteration
            )

            print("\nNEW OVERALL BEST SOLUTION FOUND")
            print(f"Best cost: {best_score:.3f}")
            print(f"Restart: {global_best_restart}")

            print(
                f"Best-cost improvement iteration "
                f"within that restart: "
                f"{global_best_iteration}"
            )

    print("\n" + "=" * 60)
    print("STABILITY SUMMARY")
    print("=" * 60)

    print(
        f"Stability window: {STABLE_WINDOW} iterations"
    )

    for result in stability_results:
        print(
            f"Restart {result['restart']:2d} | "
            f"Best cost: {result['best_cost']:.3f} | "
            f"Last improvement: "
            f"{result['last_improvement_iteration']} | "
            f"Stable at: "
            f"{result['stable_iteration']}"
        )

    print("\nOVERALL BEST SOLUTION")
    print(f"Best cost: {best_score:.3f}")
    print(f"Found during restart: {global_best_restart}")

    print(
        f"Last improvement iteration in that restart: "
        f"{global_best_iteration}"
    )

    return best_order, best_score


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            "Put scrambled_lena.mat beside this script."
        )

    data = scipy.io.loadmat(INPUT_FILE)

    if "Iscrambled" not in data:
        raise KeyError(
            "Expected a variable named 'Iscrambled' "
            "in the .mat file."
        )

    image = np.asarray(data["Iscrambled"])

    expected = TILE_SIZE * GRID

    if image.shape != (expected, expected):
        raise ValueError(
            f"Expected image shape {(expected, expected)}, "
            f"but received {image.shape}"
        )

    tiles = [
        image[
            r * TILE_SIZE:(r + 1) * TILE_SIZE,
            c * TILE_SIZE:(c + 1) * TILE_SIZE
        ]
        for r in range(GRID)
        for c in range(GRID)
    ]

    print("Image loaded successfully.")
    print(f"Image dimensions: {image.shape}")
    print(f"Number of tiles: {len(tiles)}")
    print(
        f"Tile dimensions: {TILE_SIZE} x {TILE_SIZE}"
    )

    print("\nPrecomputing edge mismatch costs...")

    horizontal, vertical = edge_costs(tiles)

    print("Edge-cost matrices calculated.")
    print("\nStarting Simulated Annealing...")

    order, score = simulated_annealing(
        horizontal, vertical
    )

    reconstructed = np.zeros_like(image)

    for position, tile_id in enumerate(order):
        r, c = divmod(position, GRID)

        reconstructed[
            r * TILE_SIZE:(r + 1) * TILE_SIZE,
            c * TILE_SIZE:(c + 1) * TILE_SIZE
        ] = tiles[int(tile_id)]

    output_dir = INPUT_FILE.parent

    png_path = output_dir / "lena_reconstructed.png"
    mat_path = output_dir / "lena_reconstructed.mat"

    Image.fromarray(reconstructed).save(png_path)

    scipy.io.savemat(
        mat_path,
        {
            "Isolved": reconstructed,
            "tile_order_0based": order.reshape(
                GRID, GRID
            ),
            "edge_mismatch_cost": np.array(
                [[score]]
            )
        }
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Reconstruction saved to {png_path}")
    print(f"MATLAB output saved to {mat_path}")

    print("\nTile order (0-based IDs):")
    print(order.reshape(GRID, GRID))

    print(
        f"\nTotal edge-mismatch cost: {score:.3f}"
    )


if __name__ == "__main__":
    main()
