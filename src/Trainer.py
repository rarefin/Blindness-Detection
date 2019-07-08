import time
import torch
import copy
import os
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from torch.optim import lr_scheduler


def train(model, dataloaders, criterion, optimizer, config):
    epoch_start = config["training"]["epoch_start"]
    num_epochs = config["training"]["num_epochs"]
    last_checkpoint_path = config["training"]["last_checkpoint_path"]

    log_file_path = config["paths"]["log_file_path"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if epoch_start > 0:
        pretrained_dict = torch.load(last_checkpoint_path)
        model.load_state_dict(pretrained_dict)

    # Slect the device: gpu or cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    criterion.to(device)

    log_file = open(log_file_path, "a")
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=5)
    start = time.time()
    for epoch in range(epoch_start, epoch_start + num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        log_file.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        log_file.write('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            preds = []
            actuals = []
            # Iterate over data.
            for X_batch, y_batch in dataloaders[phase]:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                if phase == 'train':
                    loss.backward()
                    #                         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                pred = np.argmax(outputs.detach().cpu().numpy(), 1)
                preds += list(pred)
                actuals += list(y_batch.detach().cpu().numpy())

            loss = running_loss / len(dataloaders[phase].dataset)
            preds = np.array(preds)
            actuals = np.array(actuals)
            f1_score = f1_score(actuals, preds)
            accuracy = accuracy_score(actuals, preds)
            kappa = cohen_kappa_score(actuals, preds, labels=[0, 1, 2, 3, 4])
            line = '{} Loss: {:.8f}, Accuracy: {:.3}, F1: {:.3}, Kappa: {:.3}'.format(phase, loss, accuracy, f1_score, kappa)
            print(line)
            log_file.write(line + '\n')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoint_dir + '/epoch_{}.pth'.format(epoch))
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    log_file.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    log_file.close()

    return model
