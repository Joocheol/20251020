"""Compute an approximate efficient frontier from a CSV of historical prices.

The script expects a CSV file with a three-level header matching the format of
``temp.csv`` in this repository. It extracts the close prices for every listed
asset, computes daily returns, annualises the statistics, and evaluates a grid
of long-only portfolios to approximate the efficient frontier.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


TRADING_DAYS_PER_YEAR = 252


@dataclass
class FrontierPoint:
    """Container for a single efficient frontier portfolio."""

    annual_return: float
    annual_volatility: float
    weights: Dict[str, float]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="temp.csv",
        help="Path to the input CSV file with price history.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help=(
            "Weight increment used to enumerate portfolios. Smaller values "
            "increase accuracy but also computation time."
        ),
    )
    parser.add_argument(
        "--output",
        default="efficient_frontier.csv",
        help="Where to write the efficient frontier table.",
    )
    return parser.parse_args()


def read_close_prices(path: str) -> Tuple[List[str], Dict[str, List[float]]]:
    """Read closing prices for each ticker from the CSV file."""

    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header_level_1 = next(reader)
            header_level_2 = next(reader)
            _ = next(reader)  # the third header row only contains the "Date" label
        except StopIteration as exc:  # pragma: no cover - guards against empty file
            raise ValueError("CSV file does not contain enough header rows") from exc

        tickers: List[str] = []
        close_columns: List[int] = []
        for index, (field_type, ticker) in enumerate(
            zip(header_level_1, header_level_2)
        ):
            if index == 0:
                continue  # first column is the date label
            if field_type.strip().lower() == "close":
                tickers.append(ticker.strip())
                close_columns.append(index)

        if not tickers:
            raise ValueError("No closing price columns found in the CSV header")

        price_history: Dict[str, List[float]] = {ticker: [] for ticker in tickers}
        for row in reader:
            if not row or not row[0].strip():
                continue

            try:
                prices = [row[column].strip() for column in close_columns]
            except IndexError as exc:
                raise ValueError(f"Row is missing expected columns: {row}") from exc

            if any(value == "" for value in prices):
                # Skip sessions where one of the assets did not trade.
                continue

            try:
                numeric_prices = [float(value) for value in prices]
            except ValueError as exc:
                raise ValueError(f"Failed to parse prices on row {row}") from exc

            for ticker, price in zip(tickers, numeric_prices):
                price_history[ticker].append(price)

    return tickers, price_history


def compute_simple_returns(prices: Sequence[float]) -> List[float]:
    if len(prices) < 2:
        raise ValueError("At least two prices are required to compute returns")
    returns: List[float] = []
    for previous, current in zip(prices[:-1], prices[1:]):
        if previous == 0:
            raise ValueError("Encountered zero price when computing returns")
        returns.append((current / previous) - 1.0)
    return returns


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute the mean of an empty sequence")
    return sum(values) / float(len(values))


def covariance(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Covariance expects sequences of equal length")
    if len(xs) < 2:
        raise ValueError("At least two observations are required for covariance")
    mean_x = mean(xs)
    mean_y = mean(ys)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (len(xs) - 1)


def portfolio_variance(weights: Sequence[float], covariance_matrix: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            total += wi * wj * covariance_matrix[i][j]
    return total


def iter_weight_vectors(asset_count: int, step: float) -> Iterable[Tuple[float, ...]]:
    if not 0 < step <= 1:
        raise ValueError("Step size must be in the range (0, 1]")
    increments = round(1 / step)
    if not math.isclose(step * increments, 1.0, rel_tol=1e-9):
        raise ValueError("Step size must divide 1.0 exactly for the enumeration to work")

    def rec(index: int, remaining: int, partial: List[int]) -> Iterable[Tuple[float, ...]]:
        if index == asset_count - 1:
            partial.append(remaining)
            yield tuple(value / increments for value in partial)
            partial.pop()
            return
        for value in range(remaining + 1):
            partial.append(value)
            yield from rec(index + 1, remaining - value, partial)
            partial.pop()

    yield from rec(0, increments, [])


def build_efficient_frontier(
    tickers: Sequence[str],
    expected_returns: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
    step: float,
) -> List[FrontierPoint]:
    points: List[Tuple[float, float, Tuple[float, ...]]] = []
    for weights in iter_weight_vectors(len(tickers), step):
        exp_return = sum(w * r for w, r in zip(weights, expected_returns))
        variance = portfolio_variance(weights, covariance_matrix)
        variance = max(variance, 0.0)
        volatility = math.sqrt(variance)
        points.append((exp_return, volatility, weights))

    points.sort(key=lambda item: item[0])  # sort by expected return

    efficient: List[FrontierPoint] = []
    best_volatility = float("inf")
    for exp_return, volatility, weights in points:
        if volatility + 1e-12 < best_volatility:
            best_volatility = volatility
            efficient.append(
                FrontierPoint(
                    annual_return=exp_return,
                    annual_volatility=volatility,
                    weights={ticker: weight for ticker, weight in zip(tickers, weights)},
                )
            )
    return efficient


def write_frontier_to_csv(points: Sequence[FrontierPoint], tickers: Sequence[str], path: str) -> None:
    fieldnames = ["annual_return", "annual_volatility"] + [f"weight_{ticker}" for ticker in tickers]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            row = {
                "annual_return": f"{point.annual_return:.6f}",
                "annual_volatility": f"{point.annual_volatility:.6f}",
            }
            for ticker in tickers:
                row[f"weight_{ticker}"] = f"{point.weights[ticker]:.4f}"
            writer.writerow(row)


def main() -> None:
    args = parse_arguments()
    tickers, price_history = read_close_prices(args.csv)

    return_series = [compute_simple_returns(price_history[ticker]) for ticker in tickers]
    series_length = min(len(series) for series in return_series)
    aligned_returns = [series[-series_length:] for series in return_series]

    daily_expected_returns = [mean(series) for series in aligned_returns]
    daily_covariance: List[List[float]] = []
    for i, series_i in enumerate(aligned_returns):
        row: List[float] = []
        for j, series_j in enumerate(aligned_returns):
            row.append(covariance(series_i, series_j))
        daily_covariance.append(row)

    annual_expected_returns = [ret * TRADING_DAYS_PER_YEAR for ret in daily_expected_returns]
    annual_covariance = [
        [value * TRADING_DAYS_PER_YEAR for value in row]
        for row in daily_covariance
    ]

    frontier = build_efficient_frontier(tickers, annual_expected_returns, annual_covariance, args.step)
    write_frontier_to_csv(frontier, tickers, args.output)

    print(f"Analysed {len(aligned_returns[0])} daily returns for {len(tickers)} assets.")
    print(f"Generated {len(frontier)} efficient portfolios with step size {args.step}.")
    print("Top efficient portfolios:")
    for point in frontier[:10]:
        weights_text = ", ".join(f"{ticker}={weight:.2%}" for ticker, weight in point.weights.items())
        print(
            f"  Return={point.annual_return:.2%}, Volatility={point.annual_volatility:.2%}"
            f" -> {weights_text}"
        )


if __name__ == "__main__":
    main()
