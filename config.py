import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Demo')
    
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 200)')
    
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    
    parser.add_argument('--trigger_type', type=str, default='WaNetTrigger', help='trigger type')
    parser.add_argument('--ratio', type=float, default=0.1, help='ratio of training data')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batchsize')
    
    parser.add_argument('--log_path', type=str, default='./log', help='log path')
    
    parser.add_argument('--defaultTrigger', type=bool, default=False, help='model')
    
    return parser