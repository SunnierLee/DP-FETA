import logging
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import argparse
from omegaconf import OmegaConf
from utils.util import make_dir
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def run_main(config):
    processes = []
    for rank in range(config.setup.n_gpus_per_node):
        config.setup.local_rank = rank
        config.setup.global_rank = rank + \
            config.setup.node_rank * config.setup.n_gpus_per_node
        config.setup.global_size = config.setup.n_nodes * config.setup.n_gpus_per_node
        print('Node rank %d, local proc %d, global proc %d' % (
            config.setup.node_rank, config.setup.local_rank, config.setup.global_rank))
        p = mp.Process(target=setup, args=(config, main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def setup(config, fn):
    os.environ['MASTER_ADDR'] = config.setup.master_address
    os.environ['MASTER_PORT'] = '%d' % config.setup.master_port
    os.environ['OMP_NUM_THREADS'] = '%d' % config.setup.omp_n_threads
    torch.cuda.set_device(config.setup.local_rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=config.setup.global_rank,
                            world_size=config.setup.global_size)
    fn(config)
    dist.barrier()
    dist.destroy_process_group()


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def main(config):
    workdir = os.path.join(config.setup.root_folder, config.setup.workdir)

    if config.setup.mode == 'train':
        if config.setup.global_rank == 0:
            if config.setup.mode == 'train':
                make_dir(workdir)
                gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            else:
                if not os.path.exists(workdir):
                    raise ValueError('Working directoy does not exist.')
                gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'a')

            set_logger(gfile_stream)
            logging.info(config)

        if config.setup.runner == 'train_dpdm_base':
            from runners import finetune_dpp_base
            finetune_dpp_base.training(config, workdir, config.setup.mode)
        else:
            raise NotImplementedError('Runner is not yet implemented.')
    
    elif config.setup.mode == 'pretrain':
        if config.setup.global_rank == 0:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')

            set_logger(gfile_stream)
            logging.info(config)

        if config.setup.runner == 'train_dpdm_base':
            from runners import pretrain_dpp_base
            pretrain_dpp_base.training(config, workdir, config.setup.mode)
        else:
            raise NotImplementedError('Runner is not yet implemented.')

    elif config.setup.mode == 'eval':
        if config.setup.global_rank == 0:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            set_logger(gfile_stream)
            logging.info(config)

        if config.setup.runner == 'generate_base':
            from runners import generate_base
            generate_base.evaluation(config, workdir)
        else:
            raise NotImplementedError('Runner is not yet implemented.')

        if config.setup.global_rank == 0:
            logging.info("Finished!")

    elif config.setup.mode == 'sample_show':
        if config.setup.global_rank == 0:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            set_logger(gfile_stream)
            logging.info(config)

        if config.setup.runner == 'generate_base':
            from runners import generate_base
            generate_base.sample_show(config, workdir)
        else:
            raise NotImplementedError('Runner is not yet implemented.')
    elif config.setup.mode == 'fid':
        if config.setup.global_rank == 0:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            set_logger(gfile_stream)
            logging.info(config)

        if config.setup.runner == 'generate_base':
            from runners import generate_base
            generate_base.test_fid(config, workdir)
        else:
            raise NotImplementedError('Runner is not yet implemented.')


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs="*", default=["configs/mnist_28/train_eps_1.0.yaml"])
    parser.add_argument('--workdir', type=str, default='./mnist28_e1_mean_ch22_at14_n5_sig5_q0.1')
    parser.add_argument('--mode', choices=['train', 'pretrain', 'sample_show', 'fid'], required=True)
    parser.add_argument('--root_folder', default='.')
    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.config]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    config.setup.workdir = opt.workdir
    config.setup.mode = opt.mode
    config.setup.root_folder = opt.root_folder

    if config.setup.n_nodes > 1 and False:
        raise NotImplementedError('This has not been tested.')

    run_main(config)
