import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import tqdm
import logging
import time

from torchvision import datasets, transforms

from config import *
from data_loader import *
from models.wresnet import *
from models.resnet import *
from utils import progress_bar

opt = get_args().parse_args()
best_acc = 0  # best test accuracy
# logging
timestr = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
logging.basicConfig(filename=os.path.join(opt.log_path, timestr +'log.txt'), level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(opt)

# model
model = ResNet18()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4, nesterov=True)
criterion = nn.CrossEntropyLoss().cuda()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# data
CL_train_loader = get_CL_train_loader(opt)
BD_train_loader = get_BD_train_loader(opt)
CL_test_loader = get_CL_test_loader(opt)

trainloader = BD_train_loader
testloader = CL_test_loader

# Training
def train(epoch):
    logger = logging.getLogger(__name__)
    # print('\nEpoch: %d' % epoch)
    # logger.info('Epoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # print last batch
        if batch_idx == len(trainloader)-1:
            logger.info('train: Epoch: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    logger = logging.getLogger(__name__)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx == len(testloader)-1:
                logger.info('test Epoch: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(0, opt.epochs):
    
    train(epoch)
    test(epoch)
    scheduler.step()