
import yaml
import sys
from argparse import ArgumentParser
from UCF_VIT.utils.misc import calculate_load_balancing_on_the_fly
from UCF_VIT.parse import parse_config


def main():
    parser = ArgumentParser(description="")
    parser.add_argument("config", type=str, help="Path to configuration YAML file")
    args = parser.parse_args()
    conf = parse_config(args, load_balance_offline=True)

    batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(conf, VERBOSE=True)
    

if __name__ == "__main__":
    main()
