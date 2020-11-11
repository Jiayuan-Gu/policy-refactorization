import os
import os.path as osp
import sys

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.dirname(__file__))
sys.path.insert(0, _ROOT_DIR)

import numpy as np
import cv2
import time
import gym
import argparse
import FallingDigit

import torch

torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate policy')
    parser.add_argument('--env', type=str, required=True, help='environment name')
    parser.add_argument('--n-episode', type=int, default=1000, help='number of episodes')

    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--output-dir', type=str)

    subparsers = parser.add_subparsers(dest='model_type', help='type of model')
    cnn_parser = subparsers.add_parser('cnn')
    cnn_parser.add_argument('--model', type=str, required=True, help='model name')
    cnn_parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint path')
    gnn_parser = subparsers.add_parser('gnn')
    gnn_parser.add_argument('--detector-model', type=str, required=True, help='model name')
    gnn_parser.add_argument('--detector-checkpoint', type=str, required=True, help='checkpoint path')
    gnn_parser.add_argument('--gnn-model', type=str, required=True, help='model name')
    gnn_parser.add_argument('--gnn-checkpoint', type=str, required=True, help='checkpoint path')

    args = parser.parse_args()
    return args


CNN_MODELS = {}


def register_cnn():
    from refactorization.models_cnn.plain_cnn import PlainCNN
    from refactorization.models_cnn.relation_net import RelationNet
    CNN_MODELS['PlainCNN_max'] = lambda: PlainCNN(max_pooling=True)
    CNN_MODELS['RelationNet_max'] = lambda: RelationNet(max_pooling=True)


DETECTORS = {}
GNN_MODELS = {}


def register_detector():
    from space.models.space_v0 import SPACE_v0
    from space.models.space_v1 import SPACE_v1
    DETECTORS['SPACE_v0'] = lambda: SPACE_v0()
    DETECTORS['SPACE_v1'] = lambda: SPACE_v1()


def register_gnn():
    from refactorization.models_gnn.ec_net import EdgeConvNet
    GNN_MODELS['EdgeConvNet'] = lambda: EdgeConvNet()


class PolicyInterface(object):
    def get_action(self, image, **kwargs) -> (int, dict):
        # Return action (int), and endpoints (dict)
        raise NotImplementedError()


class CNNPolicyInterface(PolicyInterface):
    def __init__(self, model_name, ckpt_path, device='cpu'):
        self.model = CNN_MODELS[model_name]()

        ckpt_data = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt_data['model'])
        self.model.eval()

        self.model.to(device)
        self.device = device

    def get_action(self, image, **kwargs):
        assert image.dtype == np.uint8
        data_batch = {
            'image': torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0),
        }
        data_batch = {k: v.to(self.device) for k, v in data_batch.items()}
        outputs = self.model(data_batch)
        logits = outputs['logits'].squeeze(0)
        action = logits.argmax().item()
        return action, {}


class GNNPolicyInterface(object):
    def __init__(self, detector_name, detector_ckpt_path, gnn_name, gnn_ckpt_path, device='cpu'):
        self.detector = DETECTORS[detector_name]()
        detector_ckpt_data = torch.load(detector_ckpt_path, map_location='cpu')
        self.detector.load_state_dict(detector_ckpt_data['model'])
        self.detector.eval()

        self.gnn = GNN_MODELS[gnn_name]()
        gnn_ckpt_data = torch.load(gnn_ckpt_path, map_location='cpu')
        self.gnn.load_state_dict(gnn_ckpt_data['model'], strict=False)
        self.gnn.eval()

        self.device = device
        self.detector.to(device)
        self.gnn.to(device)

    @torch.no_grad()
    def get_action(self, image, target_shape=(128, 128), patch_size=(16, 16), det_thresh=0.1):
        if image.shape[:2] != target_shape:
            image = cv2.resize(image, target_shape)
        img_tensor = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (b, 3, h, w)
        img_tensor = img_tensor.to(self.device)

        detector_outputs = self.detector({
            'image': img_tensor,
        }, fast=True)

        # decode boxes
        boxes = detector_outputs['boxes']  # (b, A * h1 * w1, 4)
        z_pres_p = detector_outputs['z_pres_p']  # (b, A * h1 * w1)
        normalized_boxes = detector_outputs['normalized_boxes']  # (b, A * h1 * w1, 4)

        mask = z_pres_p[0] >= det_thresh
        boxes_valid = boxes[0][mask].cpu().numpy()
        normalized_boxes_valid = normalized_boxes[0][mask]
        n = len(normalized_boxes_valid)

        if n == 0:
            print('Empty detection')
            action = np.random.randint(3)
        else:
            from torch_geometric.data import Data, Batch
            from space.utils.box_utils import image_to_glimpse

            # crop image
            patches = image_to_glimpse(img_tensor, normalized_boxes_valid.unsqueeze(0), patch_size)
            edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n)], dtype=torch.long).transpose(0, 1)
            data = Data(
                x=patches,
                edge_index=edge_index.long(),
                pos=normalized_boxes_valid.float(),
                size=torch.tensor([1], dtype=torch.int64),  # indicate batch size
            )
            data_batch = Batch.from_data_list([data])
            data_batch = data_batch.to(self.device)

            gnn_outputs = self.gnn(data_batch)
            logits = gnn_outputs['logits'].squeeze(0).cpu().numpy()
            action = logits.argmax()

        return action, {'boxes': boxes_valid}


def main():
    args = parse_args()
    env_name = args.env
    env = gym.make(env_name)

    # Initialize policy interface
    if args.model_type == 'cnn':
        register_cnn()
        policy = CNNPolicyInterface(args.model, args.checkpoint)
    elif args.model_type == 'gnn':
        register_detector()
        register_gnn()
        policy = GNNPolicyInterface(args.detector_model, args.detector_checkpoint, args.gnn_model, args.gnn_checkpoint)

    # Visualization
    if args.render:
        cv2.namedWindow('game', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('game', 640, 640)
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(os.path.join(args.output_dir, f'{env_name}.mp4'), fourcc, 10, (128, 128))

    episode_total_rewards = []
    for _ in range(args.n_episode):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, endpoints = policy.get_action(state)
            # print(action)

            if args.render:
                image_to_show = state.copy()
                image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_RGB2BGR)
                cv2.imshow('game', image_to_show)
                if cv2.waitKey(0) == 27:  # Esc key to stop
                    break
            if args.save_video:
                image_to_show = state.copy()
                image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_RGB2BGR)
                video_writer.write(image_to_show)

            state, reward, done, info = env.step(action)
            total_reward += reward
        episode_total_rewards.append(total_reward)

    # print(episode_total_rewards)
    print('On {:s}: mean {:.4f}, std {:.4f}'.format(
        env_name, np.mean(episode_total_rewards), np.std(episode_total_rewards)
    ))
    if args.save_video:
        video_writer.release()


if __name__ == '__main__':
    main()
