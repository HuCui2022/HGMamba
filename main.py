import argparse
import os
import pdb
import sys
import yaml
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 出错进行屏蔽的方法
import numpy as np
import pkg_resources
import torch
import wandb

os.environ["WANDB_MODE"] = "offline"
from torch import optim
from tqdm import tqdm
import time
from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity
from loss.pose3d import jpe as calculate_jpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import acc_error as calculate_acc_err
from data.const import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS, H36M_1_DF, H36M_2_DF, \
    H36M_3_DF
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader

from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers
from utils.data import Augmenter2D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/HGMamba-base.yaml",
                        help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint',
                        help='new checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=3407, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=6, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', default=True)
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--print-log', default=True)
    opts = parser.parse_args()
    return opts


def save_arg(args):
    # save arg  new checkpoint denotes the work_dir in yaml file
    arg_dict = vars(args)
    if not os.path.exists(args.new_checkpoint):
        os.makedirs(args.new_checkpoint)
    with open('{}/config.yaml'.format(args.new_checkpoint), 'w') as f:
        f.write(f"# command line: {' '.join(sys.argv)}\n\n")
        yaml.dump(arg_dict, f)

def print_log(str, plog, work_dir, print_time=True):
    '''
    print log
    '''
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
    print(str)
    if plog:
        with open('{}/log.txt'.format(work_dir), 'a') as f:
            print(str, file=f)

def print_time():
    localtime = time.asctime(time.localtime(time.time()))
    print_log("Local current time :  " + localtime)


def train_one_epoch(args, model, train_loader, optimizer, device, losses):
    model.train()
    for x, y in tqdm(train_loader):
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                # 相对位置
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = y[..., 2] - y[:, 0:1, 0:1, 2]  # Place the depth of first frame root to be 0

        pred = model(x)  # (N, T, 17, 3)

        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)

        loss_total = loss_3d_pos + \
                     args.lambda_scale * loss_3d_scale + \
                     args.lambda_3d_velocity * loss_3d_velocity + \
                     args.lambda_lv * loss_lv + \
                     args.lambda_lg * loss_lg + \
                     args.lambda_a * loss_a + \
                     args.lambda_av * loss_av  # default : lambda_scale = 0.5, velocity=20

        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()


def evaluate(args, model, test_loader, datareader, device, epoch):
    # print("[INFO] Evaluation")
    print_log("[INFO] Evaluation", True, args.new_checkpoint)
    results_all = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)

            if args.flip:
                batch_input_flip = flip_data(x)
                predicted_3d_pos_1 = model(x)
                predicted_3d_pos_flip = model(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model(x)
            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                y[:, 0, 0, 2] = 0

            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    if args.add_velocity:
        action_clips = action_clips[:, :-1]
        factor_clips = factor_clips[:, :-1]
        frame_clips = frame_clips[:, :-1]
        gt_clips = gt_clips[:, :-1]

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    jpe_all = np.zeros((num_test_frames, args.num_joints))
    e2_all = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    results_joints = [{} for _ in range(args.num_joints)]
    results_accelaration = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        results_accelaration[action] = []
        for joint_idx in range(args.num_joints):
            results_joints[joint_idx][action] = []

    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = calculate_mpjpe(pred, gt)
        jpe = calculate_jpe(pred, gt)
        for joint_idx in range(args.num_joints):
            jpe_all[frame_list, joint_idx] += jpe[:, joint_idx]
        acc_err = calculate_acc_err(pred, gt)
        acc_err_all[frame_list[:-2]] += acc_err
        e1_all[frame_list] += err1
        err2 = calculate_p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results_procrustes[action].append(err2)
            acc_err = acc_err_all[idx] / oc[idx]
            results[action].append(err1)
            results_accelaration[action].append(acc_err)
            for joint_idx in range(args.num_joints):
                jpe = jpe_all[idx, joint_idx] / oc[idx]
                results_joints[joint_idx][action].append(jpe)
    final_result_procrustes = []
    final_result_joints = [[] for _ in range(args.num_joints)]
    final_result_acceleration = []
    final_result = []

    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
        final_result_acceleration.append(np.mean(results_accelaration[action]))
        for joint_idx in range(args.num_joints):
            final_result_joints[joint_idx].append(np.mean(results_joints[joint_idx][action]))

    joint_errors = []
    for joint_idx in range(args.num_joints):
        joint_errors.append(
            np.mean(np.array(final_result_joints[joint_idx]))
        )
    joint_errors = np.array(joint_errors)
    e1 = np.mean(np.array(final_result))
    assert round(e1, 4) == round(np.mean(joint_errors),
                                 4), f"MPJPE {e1:.4f} is not equal to mean of joint errors {np.mean(joint_errors):.4f}"
    acceleration_error = np.mean(np.array(final_result_acceleration))
    e2 = np.mean(np.array(final_result_procrustes))

    # 更加细节的骨骼错误, 只在验证和最后一次进行
    if args.eval_only or epoch == args.epochs - 1:
        print('eval_additional/upper_body_error', np.mean(joint_errors[H36M_UPPER_BODY_JOINTS]))
        print('eval_additional/lower_body_error', np.mean(joint_errors[H36M_LOWER_BODY_JOINTS]))
        print('eval_additional/1_DF_error', np.mean(joint_errors[H36M_1_DF]))
        print('eval_additional/2_DF_error', np.mean(joint_errors[H36M_2_DF]))
        print('eval_additional/3_DF_error', np.mean(joint_errors[H36M_3_DF]))

        # for joint_idx in range(args.num_joints):
        for joint_idx in range(17):
            print(f'Joint {H36M_JOINT_TO_LABEL[joint_idx]} error: {joint_errors[joint_idx]:.4f} mm')

    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Acceleration error:', acceleration_error, 'mm/s^2')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')


    print_log('Protocol #1 Error (MPJPE): ' + str(e1) + 'mm', True, args.new_checkpoint)
    print_log('Acceleration error: ' + str(acceleration_error) + 'mm/s^2', True, args.new_checkpoint)
    print_log('Protocol #2 Error (P-MPJPE): ' + str(e2) + 'mm', True, args.new_checkpoint)
    print_log('eval_additional/upper_body_error: ' + str(np.mean(joint_errors[H36M_UPPER_BODY_JOINTS])), True, args.new_checkpoint)
    print_log('eval_additional/lower_body_error: ' + str(np.mean(joint_errors[H36M_LOWER_BODY_JOINTS])), True, args.new_checkpoint)
    print_log('eval_additional/1_DF_error: ' + str(np.mean(joint_errors[H36M_1_DF])), True, args.new_checkpoint)
    print_log('eval_additional/2_DF_error: ' + str(np.mean(joint_errors[H36M_2_DF])), True, args.new_checkpoint)
    print_log('eval_additional/3_DF_error: ' + str(np.mean(joint_errors[H36M_3_DF])), True, args.new_checkpoint)
    for joint_idx in range(17):
        print_log(f'Joint {H36M_JOINT_TO_LABEL[joint_idx]} error: {joint_errors[joint_idx]:.4f} mm', True, args.new_checkpoint)

    print_log('----------', True, args.new_checkpoint)


    return e1, e2, joint_errors, acceleration_error


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)


def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')

    common_loader_params = {
        'batch_size': args.batch_size,
        # 'num_workers': 0,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        # 'prefetch_factor': (opts.num_cpus - 1) // 3,
        'prefetch_factor': 3,
        'persistent_workers': False
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    datareader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                                data_stride_train=args.n_frames // 3, data_stride_test=args.n_frames,
                                dt_root=args.data_root, dt_file=args.dt_file)  # Used for H36m evaluation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    n_params = count_param_numbers(model)
    # print(f"[INFO] Number of parameters: {n_params:,}")
    print_log(f"Number of parameters: {n_params:,}", opts.print_log, args.new_checkpoint)   # new_checkpoint is work_dir
    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr,
                            weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float('inf')  # Used for storing the best model
    best_epoch = 0
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()
    # print(opts.checkpoint)

    # this args must set in yaml file
    if args.checkpoint:
        opts.checkpoint = args.checkpoint
        # print(opts.checkpoint)
    if args.resume:
        opts.resume = args.resume
    if args.checkpoint_file:
        opts.checkpoint_file = args.checkpoint_file
    if args.eval_only:
        opts.eval_only = args.eval_only

    print(args.checkpoint, args.resume, args.checkpoint_file, args.eval_only)
    print_log(args.checkpoint, opts.print_log, args.new_checkpoint)
    print(opts.checkpoint, opts.resume, opts.checkpoint_file, opts.eval_only)
    print_log(opts.checkpoint, opts.print_log, args.new_checkpoint)

    # pdb.set_trace()  opts.checkpoint can set in config file, only used when resume and valid
    if opts.checkpoint:
        # if checkpoint, resume the training, by checkpoint with checkpointfile with .pth.tr end.
        checkpoint_path = os.path.join(opts.checkpoint,
                                       opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)
            # print("weights loaded from checkpoint, done.")
            print_log("weights loaded from checkpoint, done.", opts.print_log, args.new_checkpoint)
            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            # print("[WARN] Checkpoint path is empty. Starting from the beginning")
            print_log("[WARN] Checkpoint path is empty. Starting from the beginning", opts.print_log, args.new_checkpoint)
            opts.resume = False

    # can set in config file
    if not opts.eval_only:
        # can set in config file
        if opts.resume:
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                           project='MotionMetaFormer',
                           resume="must",
                           settings=wandb.Settings(start_method='fork'))
        else:
            print(f"Run ID: {wandb_id}")
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                           name=opts.wandb_name,
                           project='MotionMetaFormer',
                           settings=wandb.Settings(start_method='fork'))
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args)
                installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
                wandb.config.update({'installed_packages': installed_packages})

    # checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    # checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')     # new_checkpoint with time
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # checkpoint_path_latest = os.path.join(opts.new_checkpoint, timestr + 'latest_epoch.pth.tr')   # if use default dir name
    checkpoint_path_latest = os.path.join(args.new_checkpoint, timestr + 'latest_epoch.pth.tr')   # set the work_dir in yaml file
    # checkpoint_path_best = os.path.join(opts.new_checkpoint, timestr + 'best_epoch.pth.tr')  # new_checkpoint with time
    checkpoint_path_best = os.path.join(args.new_checkpoint, timestr + 'best_epoch.pth.tr')  # set the work_dir in yaml file


    for epoch in range(epoch_start, args.epochs):
        # only evaluate the model on test set
        if opts.eval_only:
            # print("[INFO] Evaluation only")
            print_log("[INFO] Evaluation only", opts.print_log, args.new_checkpoint)
            evaluate(args, model, test_loader, datareader, device, epoch)
            # print("[INFO] Evaluation done")
            print_log("[INFO] Evaluation done", opts.print_log, args.new_checkpoint)
            exit()

        # print(f"[INFO] epoch {epoch}")
        print_log(f"[INFO] epoch {epoch}", opts.print_log, args.new_checkpoint)
        loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, device, losses)

        mpjpe, p_mpjpe, joints_error, acceleration_error = evaluate(args, model, test_loader, datareader, device, epoch)

        # record the best model info : epoch, mpjpe, p_mpjpe, joints_error, acceleration_error
        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
            best_epoch = epoch


        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe, wandb_id)

        joint_label_errors = {}
        for joint_idx in range(args.num_joints):
            joint_label_errors[f"eval_joints/{H36M_JOINT_TO_LABEL[joint_idx]}"] = joints_error[joint_idx]
        if opts.use_wandb:
            wandb.log({
                'lr': lr,
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/loss_3d_scale': losses['3d_scale'].avg,
                'train/loss_3d_velocity': losses['3d_velocity'].avg,
                'train/loss_2d_proj': losses['2d_proj'].avg,
                'train/loss_lg': losses['lg'].avg,
                'train/loss_lv': losses['lv'].avg,
                'train/loss_angle': losses['angle'].avg,
                'train/angle_velocity': losses['angle_velocity'].avg,
                'train/total': losses['total'].avg,
                'eval/mpjpe': mpjpe,
                'eval/acceleration_error': acceleration_error,
                'eval/min_mpjpe': min_mpjpe,
                'eval/p-mpjpe': p_mpjpe,
                'eval_additional/upper_body_error': np.mean(joints_error[H36M_UPPER_BODY_JOINTS]),
                'eval_additional/lower_body_error': np.mean(joints_error[H36M_LOWER_BODY_JOINTS]),
                'eval_additional/1_DF_error': np.mean(joints_error[H36M_1_DF]),
                'eval_additional/2_DF_error': np.mean(joints_error[H36M_2_DF]),
                'eval_additional/3_DF_error': np.mean(joints_error[H36M_3_DF]),
                **joint_label_errors
            }, step=epoch + 1)
        lr = decay_lr_exponentially(lr, lr_decay, optimizer)
        # printlog min_mpjpe, best_epoch
        print_log(f"min_mpjpe: {min_mpjpe}, best_epoch: {best_epoch}", opts.print_log, args.new_checkpoint)


    if opts.use_wandb:
        artifact = wandb.Artifact(f'model', type='model')
        artifact.add_file(checkpoint_path_latest)
        artifact.add_file(checkpoint_path_best)
        wandb.log_artifact(artifact)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)   # args come from config file
    save_arg(args)
    train(args, opts)  # config, opts


if __name__ == '__main__':
    main()
