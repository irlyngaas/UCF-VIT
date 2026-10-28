
import yaml
import sys
from argparse import ArgumentParser
from UCF_VIT.utils.misc import calculate_load_balancing_on_the_fly
from UCF_VIT.parse import parse_config


def main():
    """Compute and print the offline load-balancing assignment for a training run.

    Parses a config YAML file given as a command-line argument, then computes how
    dataset batches should be distributed across ranks/epochs so that each rank
    processes a balanced amount of data, printing the details as it goes.
    """
    parser = ArgumentParser(description="")
    parser.add_argument("config", type=str, help="Path to configuration YAML file")
    args = parser.parse_args()
    conf = parse_config(args, load_balance_offline=True)

    batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(conf, VERBOSE=True)
    

if __name__ == "__main__":
    main()
