import numpy.typing as npt
import multiprocessing

import fsdv.path_planning.raceline.minimum_curvature as minimum_curvature
import fsdv.path_planning.raceline.raceline as raceline
import fsdv.path_planning.raceline.frenet as frenet


def run_case(
    i: int,
    spline: raceline.Spline,
    splines: tuple[raceline.Spline, ...],
    lengths: tuple[float, ...],
) -> list[tuple[int, int, frenet.Projection]]:
    """
    Loops over every point from `splines` and checks:
        1. the distance from point to raceline is approximately 0
        2. progress is within the spline that the point forms
        3. the index of the spline found is the same as the index of the spline that the point forms
        (or the next spline, since every last point of a spline is the first point of the next spline)

    Parameters
    ----------
    i : int
        index of spline

    spline : raceline.Spline
        spline to minimize distance from

    splines : tuple[raceline.Spline, ...]
        splines representing raceline

    lengths : tuple[float, ...]
        prefix sum of lengths


    Returns
    -------
    list[tuple[int, int, Projection]]
        a list of:
            int : i, index of spline
            int : j, index of point within spline.points
            Projection : result from frenet


    """
    points: npt.NDArray = spline.points
    n_row: int
    n_col: int
    n_row, n_col = points.shape
    results: list[tuple[int, int, frenet.Projection]] = []
    for j in range(n_col):
        j: int
        result: frenet.Projection = frenet.frenet(
            x=points.item(j),
            y=points.item(n_col + j),
            splines=splines,
            lengths=lengths,
            prev_progress=lengths[i],
        )
        results.append((i, j, result))
    return results


def test_multiprocess(
    splines: tuple[raceline.Spline, ...], lengths: tuple[float, ...]
) -> None:
    """
    Multi core testing for frenet. For more details look at the `run_case` function.

    Parameters
    ----------
    splines : tuple[raceline.Spline, ...]

    lengths : tuple[float, ...]

    """
    test_cases: list[
        tuple[
            int, raceline.Spline, tuple[raceline.Spline, ...], tuple[float, ...]
        ]
    ] = [
        (i, spline, splines, lengths)
        for i, spline in enumerate(splines)
    ]
    with multiprocessing.Pool() as p:
        result: list[list[tuple[int, int, frenet.Projection]]] = p.starmap(
            run_case, test_cases
        )
    check(splines=splines, lengths=lengths, results=result)


def test(
    splines: tuple[raceline.Spline, ...], lengths: tuple[float, ...]
) -> None:
    """
    Single core testing for frenet. For more details look at the `run_case` function.

    Parameters
    ----------
    splines : tuple[raceline.Spline, ...]
        splines representing raceline

    lengths : tuple[float, ...]
        prefix sum of length

    """
    test_cases: list[
        tuple[
            int, raceline.Spline, tuple[raceline.Spline, ...], tuple[float, ...]
        ]
    ] = [
        (i, spline, splines, lengths)
        for i, spline in enumerate(splines)
    ]
    result: list[list[tuple[int, int, frenet.Projection]]] = []
    for test_case in test_cases:
        result.append(run_case(*test_case))
    check(splines=splines, lengths=lengths, results=result)


def check(
    splines: tuple[raceline.Spline, ...],
    lengths: tuple[float, ...],
    results: list[list[tuple[int, int, frenet.Projection]]],
) -> None:
    n: int = len(lengths)
    for res in results:
        res: list[tuple[int, int, frenet.Projection]]
        for i, j, result in res:
            i: int
            j: int
            length: float = result.progress
            index: int = result.min_index
            distance: float = result.min_distance
            if i == n - 1:
                if not (
                    (
                        (0 if i == 0 else lengths[i - 1]) <= length and length,
                        4 <= lengths[i],
                    )
                    or (j == 3 and (length == 0 or length == lengths[-1]))
                    # Edge case if we are trying last point of track
                    # it will find the first spline and not the last spline since they both include the last point of the track
                    # In this case, the length should be either 0 or lengths[-1]
                ):
                    print(
                        f"Length failed {(0 if i == 0 else lengths[i - 1])}, {length}, {lengths[i]}, {index} on {i}, {j}"
                    )
            else:
                if (
                    not (0 if i == 0 else lengths[i - 1]) <= length
                    and length <= lengths[i]
                ):
                    print(
                        f"Length failed {(0 if i == 0 else lengths[i - 1])}, {length}, {lengths[i]}, {index} on {i}, {j}"
                    )
            if j == 1 or j == 2:
                if not (index == i % n):
                    print(f"Index failed. Expected: {i % n} Found: {index} on {i}, {j}")
            if j == 0:
                if not (index == (i - 1) % n or index == i % n):
                    print(
                        f"Index failed. Expected: {(i - 1) % n} or {i % n} Found: {index} on {i}, {j}"
                    )
            if j == 3:
                if not (index == (i + 1) % n or index == i % n):
                    print(
                        f"Index failed. Expected: {(i + 1) % n} or {i % n} Found: {index} on {i}, {j}"
                    )

            if abs(distance) > 1e-4:
                print(f"minimum distance is {abs(distance)}, which is non-zero")


def main(filename: str) -> None:
    """
    Test frenet on track from `filename.csv`

    Parameters
    ----------
    filename : str

    """
    global xres
    global yres
    points = minimum_curvature.generateFromFile(filename=filename)

    xres = points[:, 0]
    yres = points[:, 1]

    splines: tuple[raceline.Spline]
    lengths: tuple[float]
    splines, lengths = raceline.raceline_gen(points)

    # Testing all points
    test(splines=splines, lengths=lengths)
    # test_multiprocess(splines=splines, lengths=lengths)

    # Profiling
    # import cProfile, pstats
    #
    # cProfile.runctx(
    #     "test(splines=splines, lengths=lengths)",
    #     globals=globals(),
    #     locals=locals(),
    #     filename="restats",
    # )
    # p = pstats.Stats("restats")
    # p.strip_dirs().sort_stats("tottime").print_stats()


if __name__ == "__main__":
    main("Austin.csv")
