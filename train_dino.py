# -*- coding: utf-8 -*-

# 주요 라이브러리 임포트
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import warnings
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler

# 사용자 정의 모듈 임포트
from optimizers.make_optimizer import make_optimizer
from models.model import make_model
from datasets.make_dataloader import make_dataset
from tool.utils_server import save_network, copyfiles2checkpoints
from losses.triplet_loss import Tripletloss
from losses.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss
from losses.dino_distill_loss import dino_cls_distill_loss

warnings.filterwarnings("ignore")
version = torch.__version__

# 학습 옵션 파서 정의

def get_parse():
    parser = argparse.ArgumentParser(description='Training with DINO-style Self-Distillation')
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--data_dir', default="/workspace/mount/SSD_2T_a/AAM_Data/University_dataset/University-Release/train", type=str)
    parser.add_argument('--train_all', action='store_true', help='use all training data' )
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
    parser.add_argument('--num_worker', default=6,type=int, help='' )
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=256, type=int, help='height')
    parser.add_argument('--w', default=256, type=int, help='width')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--num_epochs', default=120, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--moving_avg', default=0.996, type=float)  # EMA 이동 평균 비율
    parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
    parser.add_argument('--block', default=1, type=int, help='')
    parser.add_argument('--views', default=2, type=int)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--autocast', action='store_true', default=True)
    parser.add_argument('--distill_start_epoch', default=10, type=int, help='self-distillation 시작 epoch')
    parser.add_argument('--steps', default=[70,110], type=int, help='' )
    parser.add_argument('--backbone', default="VIT-S", type=str, help='' )
    parser.add_argument('--sample_num', default=1, type=float, help='num of repeat sampling' )
    parser.add_argument('--pretrain_path', default="/workspace/FSRA/pretrain_model/vit_small_p16_224-15ec54c9.pth", type=str, help='' )
    return parser.parse_args()

# Teacher 모델을 EMA 방식으로 업데이트

def update_ema(student_model, teacher_model, alpha):
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data = alpha * t_param.data + (1. - alpha) * s_param.data

# 2 단계 용임.
def compute_similarity_matrix(feat):  # feat: (B, N, D)
    feat_norm = F.normalize(feat, dim=-1)
    sim_matrix = torch.matmul(feat_norm, feat_norm.transpose(1, 2))  # (B, N, N)
    return sim_matrix

# 주요 학습 함수 정의

def train_model(student_model, teacher_model, opt, optimizer, scheduler, dataloaders, dataset_sizes):
    use_gpu = opt.use_gpu
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    distill_criterion = nn.MSELoss()  # self-distillation용 loss

    for epoch in range(opt.num_epochs):
        student_model.train()
        teacher_model.eval()

        running_loss = 0.0
        # 데이터 반복
        for data, data2, data3 in dataloaders:
            inputs, labels = data
            inputs2, labels2 = data2
            inputs3, labels3 = data3
            if inputs.size(0) < opt.batchsize:
                continue

            if use_gpu:
                inputs = inputs.cuda()
                inputs3 = inputs3.cuda()
                labels = labels.cuda()
                labels3 = labels3.cuda()

            optimizer.zero_grad()

            # AMP(Mixed Precision) + DINO-style distillation
            with autocast():
                student_outputs, student_outputs2 = student_model(inputs, inputs3)
                with torch.no_grad():
                    teacher_outputs, teacher_outputs2 = teacher_model(inputs, inputs3)

                # Classification loss 계산
                cls_loss = cal_loss(student_outputs[0], labels, criterion) + cal_loss(student_outputs2[0], labels3, criterion)

                # self-distillation은 일정 epoch 이후부터 적용
                if epoch >= opt.distill_start_epoch:
                    distill_loss = dino_cls_distill_loss(student_outputs[1], teacher_outputs[1], loss_type='mse')
                else:
                    distill_loss = torch.tensor(0.0, device=cls_loss.device)

                total_loss = cls_loss + distill_loss

            # 역전파 및 최적화
            if opt.autocast:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # Teacher 모델 EMA 업데이트
            update_ema(student_model, teacher_model, opt.moving_avg)
            running_loss += total_loss.item() * inputs.size(0)

        # 에폭별 결과 출력 및 스케줄러 업데이트
        print(f"Epoch {epoch}/{opt.num_epochs - 1}, Loss: {running_loss / dataset_sizes['satellite']:.4f}, Distill: {'ON' if epoch >= opt.distill_start_epoch else 'OFF'}")
        scheduler.step()

        # 모델 저장
        if epoch % 10 == 9:
            save_network(student_model, opt.name, epoch)

# 전체 파이프라인 실행

def main():
    opt = get_parse()
    opt.use_gpu = torch.cuda.is_available()

    if opt.use_gpu:
        torch.cuda.set_device(int(opt.gpu_ids.split(',')[0]))

    # 데이터 및 클래스 로딩
    dataloaders, class_names, dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)

    # 모델 정의 (Student + Teacher 복사)
    student_model = make_model(opt)
    teacher_model = copy.deepcopy(student_model)
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = student_model.cuda()
    teacher_model = teacher_model.cuda()

    optimizer, scheduler = make_optimizer(student_model, opt)
    copyfiles2checkpoints(opt)

    # 학습 수행
    train_model(student_model, teacher_model, opt, optimizer, scheduler, dataloaders, dataset_sizes)

if __name__ == '__main__':
    main()
