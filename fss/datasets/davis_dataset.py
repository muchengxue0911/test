import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from fss.datasets.range_transform import im_normalization, im_mean
from fss.datasets.reseed import reseed


class DAVISDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl

        self.videos = []
        self.frames = {}
        self.nclass = 1

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 3:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))
        print(vid_list)
        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
        else:

            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
            ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        # trials = 0
        # while trials < 5:
        info['frames'] = [] # Appended with actual frames

        # Don't want to bias towards beginning/end
        this_max_jump = min(len(frames), self.max_jump)
        start_idx = np.random.randint(len(frames)-this_max_jump+1)
        f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
        f1_idx = min(f1_idx, len(frames)-this_max_jump, len(frames)-1)

        f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
        f2_idx = min(f2_idx, len(frames)-this_max_jump//2, len(frames)-1)

        frames_idx = [start_idx, f1_idx, f2_idx]
        if np.random.rand() < 0.5:
            # Reverse time
            frames_idx = frames_idx[::-1]

        sequence_seed = np.random.randint(2147483647)
        images = []
        masks = []

        frame0_name = frames[start_idx][:-4] + '.jpg'
        png0_name = frames[start_idx][:-4] + '.png'

        frame1_name = frames[f1_idx][:-4] + '.jpg'
        png1_name = frames[f1_idx][:-4] + '.png'

        # frame0 处理
        reseed(sequence_seed)
        frame0_im = Image.open(path.join(vid_im_path, frame0_name)).convert('RGB')
        frame0_im = self.all_im_dual_transform(frame0_im)
        frame0_im = self.all_im_lone_transform(frame0_im)
        reseed(sequence_seed)
        frame0_gt = Image.open(path.join(vid_gt_path, png0_name)).convert('P')
        frame0_gt = self.all_gt_dual_transform(frame0_gt)

        pairwise_seed = np.random.randint(2147483647)
        reseed(pairwise_seed)
        frame0_im = self.pair_im_dual_transform(frame0_im)
        frame0_im = self.pair_im_lone_transform(frame0_im)
        reseed(pairwise_seed)
        frame0_gt = self.pair_gt_dual_transform(frame0_gt)

        frame0_im = self.final_im_transform(frame0_im)
        frame0_gt = torch.tensor(np.array(frame0_gt))
        frame0_gt = (frame0_gt > 0).type(torch.uint8)

        # frame1 处理
        reseed(sequence_seed)
        frame1_im = Image.open(path.join(vid_im_path, frame1_name)).convert('RGB')
        frame1_im = self.all_im_dual_transform(frame1_im)
        frame1_im = self.all_im_lone_transform(frame1_im)
        reseed(sequence_seed)
        frame1_gt = Image.open(path.join(vid_gt_path, png1_name)).convert('P')
        frame1_gt = self.all_gt_dual_transform(frame1_gt)

        pairwise_seed = np.random.randint(2147483647)
        reseed(pairwise_seed)
        frame1_im = self.pair_im_dual_transform(frame1_im)
        frame1_im = self.pair_im_lone_transform(frame1_im)
        reseed(pairwise_seed)
        frame1_gt = self.pair_gt_dual_transform(frame1_gt)

        frame1_im = self.final_im_transform(frame1_im)
        frame1_gt = torch.tensor(np.array(frame1_gt))
        frame1_gt = (frame1_gt > 0).type(torch.uint8)

        PASCAL_CLASSNAMES = [
            'skate-park', 'motocross-bumps', 'drift-straight', 'soapbox', 'parkour', 'shooting', 'drift-chicane', 'bike-packing',
            'elephant', 'tuk-tuk', 'lab-coat', 'horsejump-high', 'drift-turn', 'dog', 'car-turn', 'hike',
            'breakdance-flare', 'bmx-bumps', 'surf', 'scooter-board', 'boxing-fisheye', 'cows', 'dance-jump', 'lucia',
            'libby', 'crossing', 'drone', 'koala', 'dogs-scale', 'rhino', 'india', 'cat-girl',
            'boat', 'rollerblade', 'miami-surf', 'car-roundabout', 'dancing', 'snowboard', 'dance-twirl', 'car-shadow',
            'loading', 'scooter-gray', 'stunt', 'flamingo', 'mallard-water', 'scooter-black', 'bear', 'color-run',
            'planes-water', 'pigs', 'tractor-sand', 'mbike-trick', 'lindy-hop', 'varanus-cage', 'schoolgirls', 'longboard',
            'gold-fish', 'classic-car', 'kite-walk', 'stroller', 'breakdance_short', 'judo', 'train', 'sheep',
            'single', 'mallard-fly', 'bmx-trees', 'walking', 'breakdance', 'bus', 'motocross-jump', 'hockey',
            'swing', 'disc-jockey', 'dogs-jump', 'goat', 'camel', 'kite-surf', 'horsejump-low', 'tennis',
            'dog-agility', 'kid-football', 'paragliding', 'upside-down', 'paragliding-launch', 'motorbike', 'rallye', 'blackswan',
            'dog-gooses', 'lady-running', 'soccerball', 'night-race']

        classes = torch.tensor([[PASCAL_CLASSNAMES.index(video)]], dtype=torch.int64).reshape(1, 1)

        output = {'query_images': frame1_im[None], 'query_segmentations': frame1_gt[None],
                  'support_images': frame0_im[None], 'support_segmentations': frame0_gt[None],
                  'query_classes': classes, 'support_classes': classes,
                  'identifier': f"class {video}, query {idx}"}


        return output

    def __len__(self):
        return len(self.videos)