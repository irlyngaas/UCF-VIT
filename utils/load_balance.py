
import yaml
import sys
from UCF_VIT.utils.misc import calculate_load_balancing_on_the_fly


def main():
    config_path = sys.argv[1]
    num_total_ranks = int(sys.argv[2])

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    fsdp_size = conf['parallelism']['fsdp_size']

    simple_ddp_size = conf['parallelism']['simple_ddp_size']

    tensor_par_size = conf['parallelism']['tensor_par_size']

    data_par_size = fsdp_size * simple_ddp_size
    num_total_ddp_ranks = data_par_size

    assert data_par_size * tensor_par_size == num_total_ranks, "Need to use all ranks"

    batch_size = conf['data']['batch_size']

    batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(config_path, num_total_ddp_ranks, batch_size, VERBOSE=True)
    

if __name__ == "__main__":
    main()
