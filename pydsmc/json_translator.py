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
    log_dir: str | os.PathLike | bytes,
    *,
    include_timeseries: bool = False,
    first_converged_row: bool = False,
    only_latest: bool = True,
):
    settings_path = Path(log_dir) / "settings.json"
    with settings_path.open("r") as f:
        eval_settings = json.load(f)

    with settings_path.open() as f_setting:
        set_data = json.load(f_setting)

    prop_ids = eval_settings["property_ids"]

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
                conv = False
                for line in f_results:
                    res_data = json.loads(line)
                    if res_data["intv_converged"]:
                        last_results.append(res_data | set_data | pset_data)
                        conv = True
                        break
                if not conv:
                    last_results.append(res_data | set_data | pset_data)

            else:
                conv = -1
                lines = f_results.readlines()
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
    log_dir: str | os.PathLike | bytes,
    save_path: str | os.PathLike | bytes | None = None,
    *,
    include_timeseries: bool = False,
    first_converged_row: bool = False,
    save: bool = True,
    only_latest: bool = True,
):
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
