# coding=utf-8
import os
import math
import pickle
from unittest import result
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torchvision.io import read_video
import random
import utils.logging as logging
from datasets.data_augment import create_data_augment, create_ssl_data_augment

logger = logging.get_logger(__name__)

PENN_ACTION_LIST = [
    'baseball_pitch',
    'baseball_swing',
    'bench_press',
    'bowl',
    'clean_and_jerk',
    'golf_swing',
    'jumping_jacks',
    'pushup',
    'pullup',
    'situp',
    'squat',
    'tennis_forehand',
    'tennis_serve'
]

class PennAction(torch.utils.data.Dataset):
    def __init__(self, cfg, split, dataset_name=None, mode="auto", sample_all=False):
        assert split in ["train", "val", "test"]
        self.cfg = cfg
        self.split = split
        if mode == "auto":
            self.mode = "train" if self.split=="train" else "eval"
        else:
            self.mode = mode
        self.sample_all = sample_all
        self.num_contexts = cfg.DATA.NUM_CONTEXTS

        with open(os.path.join(cfg.PATH_TO_DATASET, split+'.pkl'), 'rb') as f:
            self.dataset, self.action_to_indices = pickle.load(f)

        if dataset_name is not None:
            indices = self.action_to_indices[PENN_ACTION_LIST.index(dataset_name)]
            self.dataset = [self.dataset[index] for index in indices]
            logger.info(f"{len(self.dataset)} {self.split} samples of {dataset_name} dataset have been read.")
        else:
            logger.info(f"{len(self.dataset)} {self.split} samples of Penn Action dataset have been read.")
        if not self.sample_all:
            seq_lens = [data['seq_len'] for data in self.dataset]
            hist, bins = np.histogram(seq_lens, bins='auto')
            print(list(bins.astype(np.int)))
            print(list(hist))

        self.num_frames = cfg.TRAIN.NUM_FRAMES
        self.num_segments = cfg.TRAIN.NUM_SEGMENTS
        self.overlap_rate = cfg.TRAIN.OVERLAP_RATE
        # Perform data-augmentation
        if self.cfg.SSL and self.mode=="train":
            self.data_preprocess = create_ssl_data_augment(cfg, augment=True)
        elif self.mode=="train":
            self.data_preprocess = create_data_augment(cfg, augment=True)
        else:
            self.data_preprocess = create_data_augment(cfg, augment=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        id = self.dataset[index]["id"]
        
        name = self.dataset[index]["name"]
        frame_label = self.dataset[index]["frame_label"]
        seq_len = self.dataset[index]["seq_len"]
        video_file = os.path.join(self.cfg.PATH_TO_DATASET, self.dataset[index]["video_file"])
        video, _, info = read_video(video_file, pts_unit='sec')
        video = video.permute(0,3,1,2).float() / 255.0 # T H W C -> T C H W, [0,1] tensor

        if self.cfg.DATA.ANNOTATION == 'raw':
            steps = self.sample_raw(seq_len, self.num_frames)
            bridges = self.bridge_construct(steps, self.overlap_rate, self.num_segments)
        elif self.cfg.DATA.ANNOTATION == 'phase':
            steps, bridges = self.sample_phase(seq_len, self.num_frames,frame_label)
        else:
            steps, bridges = self.sample_phase(seq_len, self.num_frames,frame_label)

        labels = frame_label[steps.long()]
        video = video[steps.long()]
        video = self.data_preprocess(video)
        return video, labels, torch.tensor(seq_len), bridges, name, id

    def sample_raw(self, seq_len, num_frames):
        if seq_len >= num_frames:
            steps = torch.randperm(seq_len) 
            steps = torch.sort(steps[:num_frames])[0] #0-value,1-index
        else:
            steps = torch.randint(low=0, high=seq_len, size=(num_frames,))
            steps = torch.sort(steps)[0] 
        return steps
    
    def sample_phase(self, seq_len, num_frames, frame_label):
        frame_list = frame_label.tolist()
        groups = list(set(frame_list))  # 获取所有组
        phase_len = []
        steps_v = []
        for group in groups:
            start = frame_list.index(group)  # 组开始的位置
            end = len(frame_list) - frame_list[::-1].index(group) -1  # 组结束的位置
            cur_len = end - start + 1
            if cur_len < 3: 
                continue
                print(frame_label,group,start,end)
                raise ValueError('The length of this video is too short:{}'.format(cur_len))
            phase_len.append(cur_len)
            if cur_len >= num_frames:
                steps = torch.tensor([start,end])
                while len(steps) < num_frames:
                    num = torch.randint(start+1, end, (1,))
                    if num not in steps:
                        steps = torch.cat((steps,num))
                steps = torch.sort(steps)[0]
            else:
                steps = torch.randint(start+1, end, size=(num_frames,))
                steps = torch.sort(steps)[0]  
            steps_v.append(steps)
        inx = random.randint(0, len(steps_v)-1)  #[]
        bridge_point = random.randint(1,num_frames-2)
        bridge = torch.tensor([[0,bridge_point,num_frames-1]])

        return steps_v[inx],bridge

    def bridge_construct(self, steps, overlap_rate, num_segments):
        avg_len = int(len(steps) / num_segments)
        half_lap = int(avg_len*overlap_rate / (2-overlap_rate))
        bridges = torch.empty(num_segments,3)
        for i in range(num_segments):
            if i == 0:
                bridge_head = random.randint(0,half_lap)  # include a,b
                bridge_tail = avg_len + random.randint(0,half_lap)
            elif i == num_segments-1:
                bridge_head = i * avg_len - random.randint(0,half_lap)
                bridge_tail =  len(steps) - random.randint(1,half_lap)
            else:
                bridge_head = i * avg_len - random.randint(0,half_lap)
                bridge_tail = (i+1) * avg_len + random.randint(0,half_lap)
            bridge_point = bridge_head + random.randint(1,avg_len-half_lap-1)
            bridges[i] = torch.tensor([bridge_head,bridge_point,bridge_tail])
            # bridges.append([bridge_head,bridge_point,bridge_tail])
        return bridges


    def bridge_phase(self,frame_label):
        bridges = torch.empty(0,3)
        frame_list = frame_label.tolist()
        groups = list(set(frame_list))  # 获取所有组
        for group in groups:
            start = frame_list.index(group)  # 组开始的位置
            end = len(frame_list) - frame_list[::-1].index(group) -1  # 组结束的位置
            if end - start <= 1: continue
            bridge_point = random.randint(start+1,end-1)
            cur = torch.tensor([[start,bridge_point,end]])
            bridges = torch.cat((bridges,cur),dim=0)

        inx = random.randint(0, len(bridges)-2)
        # print(bridges[inx],bridges[inx+1])
        assert len(bridges)>1
        return bridges[inx:inx+2]



class ActionBatchSampler(torch.utils.data.Sampler):
    def __init__(self, cfg, dataset, batch_size, shuffle=True, seed=0):
        self.dist = True if cfg.NUM_GPUS > 1 else False
        self.dataset = dataset
        self.action_to_indices = dataset.action_to_indices
        self.batch_size = batch_size
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        batch = []
        action = -1
        for i in iter(range(self.num_samples)):
            if action == -1:
                action = torch.randint(high=len(self.action_to_indices), size=(1,), dtype=torch.int64)
                indices = self.action_to_indices[action.item()]
                indices_shuffle = torch.randperm(len(indices)).tolist()
            idx = indices[indices_shuffle[len(batch)]]
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                action = -1
        
    def __len__(self):
        return self.num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch