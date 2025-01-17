import torch, torchvision
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.autograd.profiler as profiler
from torch.utils import mkldnn as mkldnn_utils
from torch.autograd.profiler import record_function
from collections import OrderedDict
import time, subprocess, os, argparse

models.__dict__['resnext101'] = models.resnext101_32x8d
models.__dict__['mnasnet_a1'] = models.mnasnet.mnasnet1_0

from mobilenet_v1 import mobilenet_v1
models.__dict__['mobilenet_v1'] = mobilenet_v1

from mobilenet import MobileNetV2
models.__dict__['mobilenet_v2'] = MobileNetV2

from shufflenet import ShuffleNet
models.__dict__['shufflenet'] = ShuffleNet

from unet2d import UNet
models.__dict__['unet'] = UNet

from unet3d import UNet3D
models.__dict__['unet3d'] = UNet3D

nDryRuns = 5 # nb of warmup steps

def benchmark():
    # benchmark settings
    parser = argparse.ArgumentParser(description='PyTorch Convnet Benchmark')
    parser.add_argument('--arch',  action='store', default='all',
                       choices=['alexnet', 'vgg11', 'inception_v3', 'resnet18', 'resnet50', 'resnext101', 'wide_resnet50_2', 'mnasnet_a1', 'mnasnet0_5',\
                            'squeezenet1_0', 'densenet121', 'mobilenet_v1', 'mobilenet_v2', 'mobilenet_v3_large', 'shufflenet', 'efficientnet_b7', 'unet', 'unet3d', 'all'],
                       help='model name can be specified. all is default.' )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='disable CUDA')
    parser.add_argument('--mkldnn', action='store_true', default=False,
                       help='use mkldnn blocked memory format')
    parser.add_argument('--channels_last', action='store_true', default=False,
                       help='use channels_last (NHWC) memory format')
    parser.add_argument('--inference', action='store_true', default=False,
                       help='run inference only')
    parser.add_argument('--single-batch-size', action='store_true', default=False,
                       help='single batch size')
    parser.add_argument('--profile', action='store_true', default=False,
                       help='enable autograd profiler')
    parser.add_argument('--collect-execution-graph', action='store_true', default=False,
                       help='collect execution graph')
    parser.add_argument("--batch-size", type=int, default=64,
                       help='batch size')
    parser.add_argument("--num-steps", type=int, default=50,
                       help='nb of steps in loop to average perf')
    parser.add_argument("--print-freq", type=int, default=5,
                       help='print frequency')

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    archs = OrderedDict()
    ### [batch_size, channels, width, height, support_channels_last, support_mkldnn_blocked]
    archs['alexnet'] = [args.batch_size, 3, 224, 224, True, True]
    archs['vgg11'] = [args.batch_size, 3, 224, 224, True, True]
    archs['inception_v3'] = [args.batch_size, 3, 299, 299, True, False]
    archs['resnet18'] = [args.batch_size, 3, 224, 224, True, True]
    archs['resnet50'] = [args.batch_size, 3, 224, 224, True, True]
    archs['resnext101'] = [args.batch_size, 3, 224, 224, True, True]
    archs['wide_resnet50_2'] = [args.batch_size, 3, 224, 224, True, True]
    archs['mnasnet_a1'] = [args.batch_size, 3, 224, 224, True, False]
    archs['mnasnet0_5'] = [args.batch_size, 3, 224, 224, True, False]
    archs['squeezenet1_0'] = [args.batch_size, 3, 224, 224, True, False]
    archs['densenet121'] = [args.batch_size, 3, 224, 224, True, False]
    archs['mobilenet_v1'] = [args.batch_size, 3, 224, 224, True, False]
    archs['mobilenet_v2'] = [args.batch_size, 3, 224, 224, True, False]
    archs['mobilenet_v3_large'] = [args.batch_size, 3, 224, 224, True, False]
    archs['shufflenet'] = [args.batch_size, 3, 224, 224, True, False]
    archs['efficientnet_b7'] = [args.batch_size, 3, 224, 224, True, False]
    archs['unet'] = [args.batch_size, 3, 64, 64, True, False]
    archs['unet3d'] = [6, 4, 64, 64, 64]
    archs_list = list(archs.keys())
    arch_dict = {args.arch: archs[args.arch]} if args.arch in archs_list else archs

    if args.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        cudnn.deterministic = True

        kernel = 'cudnn'
        p = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv',
                                    shell=True)
        visible_devices = [int(s) for s in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
        device_idx = min(visible_devices) if visible_devices is not None else 0
        device_name = str(p).split('\\n')[device_idx + 1]
    else:
        kernel = 'nn'
        p = subprocess.check_output('cat /proc/cpuinfo | grep name | head -n 1',
                                    shell = True)
        device_name = str(p).split(':')[1][:-3]

    print('Running on device: %s' % (device_name))
    print('Running on torch: %s' % (torch.__version__))
    print('Running on torchvision: %s\n' % (torchvision.__version__))

    def _time():
        if args.cuda:
            torch.cuda.synchronize()

        return time.time()

    for arch, config in arch_dict.items():
        # cover at least resnext for now per FB request
        # TODO:
        # 1. view() support?
        # 2. Dropout() support?
        if args.mkldnn and not (arch == 'resnext101' or arch == 'mobilenet_v1' or arch == 'mobilenet_v2' or arch == 'mnasnet_a1' or arch == 'resnet50' or arch == 'resnet18'):
            continue

        if arch == 'unet3d':
            batch_size, c, d, h, w = config[0], config[1], config[2], config[3], config[4]
            batch_size = 1 if args.single_batch_size else batch_size
            print('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%dx%d' %
                 (arch, kernel, batch_size, c, d, h, w))
            data = torch.randn(batch_size, c, d, h, w)
        else:
            batch_size, c, h, w = config[0], config[1], config[2], config[3]
            batch_size = 64 if arch == 'resnet50' and args.inference else batch_size
            batch_size = 1 if args.single_batch_size else batch_size
            print('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%d' %
                 (arch, kernel, batch_size, c, h, w))
            data = torch.randn(batch_size, c, h, w)

        support_channels_last = config[4]
        support_mkldnn_blocked = config[5]

        target = torch.arange(1, batch_size + 1).long()
        net = models.__dict__[arch]() # no need to load pre-trained weights for dummy data

        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            net.cuda()
            criterion = criterion.cuda()

        # use mkldnn blocked format
        if args.mkldnn:
            if not support_mkldnn_blocked:
                print("model: %s does not support mkldnn blocked format yet!" % (arch))
                continue

            data = data.to_mkldnn()
            if args.inference:
                net.eval()
                ### weight prepacking for inference
                net = mkldnn_utils.to_mkldnn(net)

        # use channels last format
        if args.channels_last:
            if not support_channels_last:
                print("model: %s does not support channels last format yet!" % (arch))
                continue

            data = data.to(memory_format=torch.channels_last)
            net = net.to(memory_format=torch.channels_last)

        if args.inference:
            net.eval()
        else:
            net.train()
            net.aux_logits = False

        for _ in range(nDryRuns):
            if args.inference:
                with torch.no_grad():
                    output = net(data)
            else:
                output = net(data)
                loss = output.sum() / 1e6 if 'unet' in arch else criterion(output, target)
                loss.backward()
                optimizer.step()    # Does the update
                optimizer.zero_grad()   # zero the gradient buffers

        time_fwd, time_bwd, time_upt = 0, 0, 0

        if args.cuda:
            event_1 = torch.cuda.Event(enable_timing=True)
            event_2 = torch.cuda.Event(enable_timing=True)
            event_3 = torch.cuda.Event(enable_timing=True)
            event_4 = torch.cuda.Event(enable_timing=True)

        with profiler.profile(args.profile, use_cuda=args.cuda, use_kineto=True) as prof:
            class dummy_record_function():
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc_value, traceback):
                    return False

            for idx in range(args.num_steps):
                should_print = ((idx + 1) % args.print_freq) == 0
                with record_function("## BENCHMARK ##") if args.collect_execution_graph else dummy_record_function():
                    with record_function("## Forward ##"):
                        t1 = _time()
                        if args.cuda:
                            event_1.record()
                        if args.inference:
                            with torch.no_grad():
                                output = net(data)
                        else:
                            output = net(data)
                        if args.cuda:
                            event_2.record()
                        t2 = _time()
                    time_fwd += event_1.elapsed_time(event_2) * 1.e-3 if args.cuda else (t2 - t1)
                    if not args.inference:
                        with record_function("## Backward ##"):
                            t1 = _time()
                            if args.cuda:
                                event_1.record()
                            optimizer.zero_grad()   # zero the gradient buffers
                            loss = output.sum() / 1e6 if 'unet' in arch else criterion(output, target)
                            loss.backward()
                            if args.cuda:
                                event_2.record()
                            t2 = _time()
                            if args.cuda:
                                event_3.record()
                            optimizer.step()        # updates
                            if args.cuda:
                                event_4.record()
                            t3 = _time()
                        time_bwd += event_1.elapsed_time(event_2) * 1.e-3 if args.cuda else (t2 - t1)
                        time_upt += event_3.elapsed_time(event_4) * 1.e-3 if args.cuda else (t3 - t2)
                    if should_print:
                        time_per_it = (time_fwd + time_bwd + time_upt) / (idx + 1) * 1000
                        print("Finished step {}/{}, {:.2f} ms/it".format(idx + 1, args.num_steps, time_per_it))

            time_fwd_avg = time_fwd / args.num_steps * 1000
            time_bwd_avg = time_bwd / args.num_steps * 1000
            time_upt_avg = time_upt / args.num_steps * 1000
            time_total = time_fwd_avg + time_bwd_avg + time_upt_avg

            print("Overall per-batch training time: {:.2f} ms".format(time_total))

        if args.profile:
            with open("convnet_benchmark.prof", "w") as prof_f:
                prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
            prof.export_chrome_trace("convnet_benchmark.json")

        print("%-30s %10s %10.2f (ms) %10.2f (imgs/s)" % (kernel, ':forward:',
              time_fwd_avg, batch_size*1000/time_fwd_avg ))
        print("%-30s %10s %10.2f (ms)" % (kernel, ':backward:', time_bwd_avg))
        print("%-30s %10s %10.2f (ms)" % (kernel, ':update:', time_upt_avg))
        print("%-30s %10s %10.2f (ms) %10.2f (imgs/s)" % (kernel, ':total:',
              time_total, batch_size*1000/time_total ))

if __name__ == '__main__':
    benchmark()
