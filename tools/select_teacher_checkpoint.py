import argparse
import numpy as np
import sys, os
import uuid

def parse_args():
    parser = argparse.ArgumentParser(description='abc')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
        required=True,
    )

    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--n-episodes', default=1000, type=int)
    parser.add_argument('--ckpt-dir', type=str, required=True)
    parser.add_argument('--n-ckpt', type=int, default=None)

    args = parser.parse_args()
    return args

from pathlib import Path
def all_files_in_dir(_dir):
    return list(Path(_dir).rglob('*'))

def all_ckpts_in_dir(_dir):
    return list(Path(_dir).rglob('*.pth'))

def index_of_ckpt(s):
    s = os.path.basename(s)
    s = s[:-4]
    return int(s)

def read_return_value_from_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
        s = content[-1]
        assert 'episodic_return_test' in s
        k1 = s.find('episodic_return_test')
        k2 = s.find('(std')
        x = float(s[k1+21:k2-1])
    return x

def eval_ckpt(args, ckpt_path, tmp_file_name):
    command = 'python dqn/main.py --debug --cfg {:s} env {:s} load_ckpt {:s} eval.n_episodes {:d} > {:s}'.format(
        args.config_file, args.env, str(ckpt_path), args.n_episodes, tmp_file_name
    )
    os.system(command)
    return read_return_value_from_file(tmp_file_name)

if __name__ == '__main__':
    '''
    python tools/select_teacher_checkpoint.py --cfg configs/falling_digit_rl/dqn_relation_net_eval.yml --env FallingDigitCIFAR_3-v1 --ckpt-dir 
    '''
    args = parse_args()

    path_list = all_ckpts_in_dir(args.ckpt_dir)
    path_list = sorted(path_list, key=index_of_ckpt)
    path_list.reverse()
    if args.n_ckpt is not None:
        path_list = path_list[:args.n_ckpt]
    # for path in path_list:
    #     print(path)
    # import pdb; pdb.set_trace()

    print('### Evaluation results over {:d} episodes in training set:'.format(args.n_episodes))
    tmp_file_name = 'tmp_eval_result_{:s}.txt'.format(str(uuid.uuid4()))
    eval_results = []
    for ckpt_path in path_list:
        eval_return = eval_ckpt(args, ckpt_path, tmp_file_name)
        eval_results.append(eval_return)
        ckpt_index_str = index_of_ckpt(ckpt_path)
        print('ckpt {:d}:\t{:.3f}'.format(index_of_ckpt(ckpt_path), eval_return))

    os.system('rm {:s}'.format(tmp_file_name))
    best_idx = np.argmax(eval_results)
    print('### -----------------------')
    print('Best teacher checkpoint:', str(path_list[best_idx]))
    print('Best evaluation result: {:.3f}'.format(eval_results[best_idx]))
    print('Run the following command to collect demonstration datset:')
    print('python tools/collect_demo_dataset_for_falling_digit.py --env {:s} --cfg {:s} --ckpt {:s}'.format(
        args.env, args.config_file, str(path_list[best_idx])
    ))

