#!python3

import argparse
import json
import os
from pprint import pprint

import pandas as pd


def __read_run_results(
    log_dir: str | os.PathLike | bytes,
    include_timeseries: bool = False,
    only_latest: bool = True
):
    settings_json_path = os.path.join(log_dir, 'settings.json')
    with open(settings_json_path, 'r') as f:
        eval_settings = json.load(f)

    # TODO: Instead of, or additionally to, logging the ids, we could log the directories, which would make parsing here simpler
    prop_ids = eval_settings['property_ids']

    last_results = []
    for entry in os.listdir(log_dir):
        full_path = os.path.join(log_dir, entry)
        if not os.path.isdir(full_path):
            continue

        if only_latest and not str(os.path.basename(entry)).rpartition('_')[2] in prop_ids:
            continue

        with open(os.path.join(full_path, 'settings.json'), 'r') as f_setting:
            set_data = json.load(f_setting)
        with open(os.path.join(full_path, 'results.jsonl'), 'r') as f_results:
            if include_timeseries:
                for line in f_results:
                    res_data = json.loads(line)
                    last_results.append(res_data | set_data)
            else:
                res_data = json.loads(f_results.readlines()[-1])
                last_results.append(res_data | set_data)

    return last_results


def jsons_to_df(
    log_dir: str | os.PathLike | bytes,
    include_timeseries: bool = False,
    save: bool = True,
    save_path: str | os.PathLike | bytes | None = None,
    only_latest: bool = True
):
    latest_results = __read_run_results(log_dir, include_timeseries=include_timeseries, only_latest=only_latest)
    df = pd.DataFrame(latest_results)
    if save:
        save_path = save_path or log_dir
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'results.jsonl')
        if (save_path.endswith('.csv')):
            df.to_csv(save_path, index=False)
        else:
            with open(save_path, 'w') as f:
                json.dump(latest_results, f, indent=4)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the results to a DataFrame and accumulated json')

    parser.add_argument(
        '--log-dir', '-l',
        type=str,
        default='../logs',
        help='The directory containing the logs'
    )
    parser.add_argument(
        '--save-path', '-s',
        type=str,
        default='logs',
        help='Path to save the accumulated results to. Supports path & file. Supports csv and jsonl'\
        ' Default is a results.jsonl in the log-dir'
    )
    parser.add_argument(
        '--no-save', '-n',
        action='store_false',
        help='Disable saving the results in a separate file, only returns the DataFrame'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print the DataFrame to stdout'
    )
    parser.add_argument(
        '--include-timeseries', '-t',
        action='store_true',
        help='Include all intermediate results in the dataframe'
    )
    parser.add_argument( # This is more ore less useless, since we introduced a subdir for every eval run
        '--all', '-a',
        action='store_true',
        help='(Deprecated) Read all the results, not only the latest as specified in the settings.json. '\
             'I.e., ignores property_ids specified in `<log_dir>/settings.json`'
    )
    args = parser.parse_args()

    res = jsons_to_df(
        log_dir=args.log_dir,
        include_timeseries=args.include_timeseries,
        save=args.no_save,
        save_path=args.save_path,
        only_latest=not args.all
    )

    if args.verbose:
        pprint(res)
