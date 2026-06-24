import os
import logging
import json
import argparse

from peak_extraction.config import load_config, config
from peak_extraction.pipeline import extract_peaks
from peak_extraction.results import generate_detailed_df, generate_methods_df


# Set up logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def parse_arguments():
    """ Args parser for config and visualization flag """

    parser = argparse.ArgumentParser(description='echem peak extractor')
    parser.add_argument('-c', '--config', type=str, help='Path to configuration TOML file', required=True)
    parser.add_argument('--display', default=False, action='store_true', help='Visualize results?')
    parser.add_argument('--save', default=False, action='store_true', help='Save extracted peak data?')
    return parser.parse_args()


def main():
    """ Peak extraction and visualization """

    # Parse arguments
    args = parse_arguments()
    load_config(args.config)

    # Perform the analysis
    logger.info('Extracting peaks')
    results_dict = extract_peaks()

    if args.save:
        for folder_path, (results, times) in results_dict.items():
            folder_name = os.path.basename(os.path.normpath(folder_path))
            detailed_save_file = f"{folder_name}_detailed_results"

            # json output
            json_path = os.path.join(folder_path, detailed_save_file + ".json")
            with open(json_path, "w") as outfile:
                json.dump(results[0], outfile, indent=2)
            logger.info(f"Data saved in: {json_path}")

            # csv output
            csv_path = os.path.join(folder_path, detailed_save_file + ".csv")
            df = generate_detailed_df(times=times, json_file=json_path)
            df.to_csv(csv_path, index=True)
            logger.info(f"Detailed results saved in: {csv_path}")

            methods_save_file = f"{folder_name}_methods_results"
            csv_path = os.path.join(folder_path, methods_save_file + ".csv")
            df = generate_methods_df(times=times, results=results)
            df.to_csv(csv_path, index=True)
            logger.info(f"Methods results saved in: {csv_path}")

    # Process and visualize the result
    if args.display:
        logger.info('Visualizing results')


if __name__ == '__main__':
    main()
