#!python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import os


def __read_run_results(
    log_dir: Path | str | os.PathLike,
    *,
    include_timeseries: bool = False,
    first_converged_row: bool = False,
    only_latest: bool = True,
) -> list[dict]:
    settings_path = Path(log_dir) / "settings.json"
    with settings_path.open() as f_setting:
        set_data = json.load(f_setting)

    prop_ids = set_data["property_ids"]

    last_results = []
    for entry in Path(log_dir).iterdir():
        if not entry.is_dir():
            continue

        if only_latest and entry.name.rpartition("_")[2] not in prop_ids:
            continue

        results_path = entry / "results.jsonl"
        prop_settings_path = entry / "settings.json"

        with prop_settings_path.open() as f_psetting:
            pset_data = json.load(f_psetting)

        with results_path.open() as f_results:
            if include_timeseries:
                for line in f_results:
                    res_data = json.loads(line)
                    last_results.append(res_data | set_data | pset_data)

            elif first_converged_row:
                last_row = None
                appended = False
                for line in f_results:
                    last_row = json.loads(line)
                    if last_row["intv_converged"]:
                        last_results.append(last_row | set_data | pset_data)
                        appended = True
                        break
                # No converged row found: fall back to the last available row (if any)
                if not appended and last_row is not None:
                    last_results.append(last_row | set_data | pset_data)

            else:
                lines = f_results.readlines()
                if not lines:
                    continue
                conv = -1
                for line in lines:
                    res_data = json.loads(line)
                    if res_data["intv_converged"]:
                        conv = res_data["total_episodes"]
                        break
                res_data = json.loads(lines[-1])
                res_data["converged@"] = conv
                last_results.append(res_data | set_data | pset_data)

    return last_results


def jsons_to_df(
    log_dir: Path | str | os.PathLike,
    save_path: Path | str | os.PathLike | None = None,
    *,
    include_timeseries: bool = False,
    first_converged_row: bool = False,
    save: bool = True,
    only_latest: bool = True,
) -> pd.DataFrame:
    """Aggregate the JSON-Lines logs of an evaluation run into a pandas `DataFrame`.

    Reads the per-property `results.jsonl` files written under `log_dir`, merges them with the run
    and property settings, and returns the combined results. Optionally also writes the aggregated
    results to disk.

    Args:
        log_dir: Directory containing an evaluation run's logs (the run's `settings.json` and the
            per-property subdirectories).
        save_path: Where to write the aggregated results. Defaults to `log_dir`. If it is a
            directory, a `results.jsonl` is written inside it; a `.csv` suffix writes CSV, otherwise
            JSON.
        include_timeseries: Include every intermediate result (the full time series) rather than
            just the final row per property.
        first_converged_row: Keep the first row where the interval converged (falling back to the
            last available row if none converged). Ignored if `include_timeseries` is set.
        save: Whether to write the aggregated results to `save_path`.
        only_latest: Only read the properties referenced in the run's `settings.json` (the latest
            run), rather than every property subdirectory present.

    Returns:
        A `DataFrame` with one row per property (or per intermediate result when `include_timeseries` is set).
    """
    latest_results = __read_run_results(
        log_dir,
        include_timeseries=include_timeseries,
        first_converged_row=first_converged_row,
        only_latest=only_latest,
    )
    df = pd.DataFrame(latest_results)
    if save:
        save_path = save_path or log_dir
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / "results.jsonl"
        if save_path.name.endswith(".csv"):
            df.to_csv(save_path, index=False)
        else:
            with save_path.open("w") as f:
                json.dump(latest_results, f, indent=4)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the results to a DataFrame and accumulated json",
    )

    parser.add_argument(
        "--log-dir",
        "-l",
        type=str,
        default="../logs",
        help="The directory containing the logs",
    )
    parser.add_argument(
        "--save-path",
        "-s",
        type=str,
        default="logs",
        help="Path to save the accumulated results to. Supports path & file. Supports csv and jsonl"
        " Default is a results.jsonl in the log-dir",
    )
    parser.add_argument(
        "--no-save",
        "-n",
        action="store_false",
        help="Disable saving the results in a separate file, only returns the DataFrame",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the DataFrame to stdout",
    )
    parser.add_argument(
        "--include-timeseries",
        "-t",
        action="store_true",
        help="Include all intermediate results in the dataframe",
    )
    parser.add_argument(  # This is more ore less useless, since we introduced a subdir for every eval run
        "--all",
        "-a",
        action="store_true",
        help="(Deprecated) Read all the results, not only the latest as specified in the settings.json. "
        "I.e., ignores property_ids specified in `<log_dir>/settings.json`",
    )
    args = parser.parse_args()

    res = jsons_to_df(
        log_dir=args.log_dir,
        include_timeseries=args.include_timeseries,
        save=args.no_save,
        save_path=args.save_path,
        only_latest=not args.all,
    )

    if args.verbose:
        pprint(res)
