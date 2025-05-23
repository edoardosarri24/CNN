import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from torchmetrics.classification import MulticlassAccuracy
from utilityFunction import count_layers, get_dataloader
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'

def train_epoch(model:nn.Module, dl:DataLoader, opt:torch.optim.Optimizer, evaluating_grad):
    model.train()
    losses = []
    grad_norm = 0.0
    for xs, ys in dl:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        opt.zero_grad()
        output = model(xs)
        loss = F.cross_entropy(output, ys)
        loss.backward()
        if(evaluating_grad):
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
        opt.step()
        losses.append(loss.detach())
    wandb.log({'grad_norm': grad_norm})
    return torch.mean(torch.stack(losses)).item()

def testing_acc(model:nn.Module, dataLoader:DataLoader, classi):
    model.eval()
    accuracy = MulticlassAccuracy(num_classes=classi).to(device)
    with torch.no_grad():
        for (xs, ys) in dataLoader:
            xs, ys = xs.to(device), ys.to(device)
            outputs = model(xs)
            preds = torch.argmax(outputs, dim=1)
            accuracy.update(preds, ys)
    return accuracy.compute().item()

def pipeline(model:nn.Module, ds_train, ds_test, project_name:str, run_name:str,
             epochs:int=20, lr:float=5e-4, opt=None, batch_size:int=128, early_stopping:bool=False, patience:int=5, delta_early_stopping:float=0.01, classi=10, evaluating_grad:bool=False):

    # W&B
    wandb.init(project=project_name,
        name=run_name,
        config={
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'deep': count_layers(model)
        })
    wandb.watch(model)

    #data
    dl_train = get_dataloader(ds_train, batch_size)
    dl_test = get_dataloader(ds_test, batch_size)

    # training
    model = model.to(device)
    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    best_acc = 0.0
    best_model = model.state_dict()
    epochs_wo_improvement = 0
    for epoch in tqdm(range(epochs), desc=f'Training: {run_name}', ncols=1000):
        train_loss = train_epoch(model, dl_train, opt, evaluating_grad)
        test_acc = testing_acc(model, dl_test, classi)
        wandb.log({'train_loss': train_loss,
                   'test_acc': test_acc})
        # early stopping
        if(early_stopping):
            if test_acc-best_acc>delta_early_stopping:
                best_acc = test_acc
                best_model = model.state_dict()
                epochs_wo_improvement = 0
            else:
                epochs_wo_improvement +=1
                if(epochs_wo_improvement>=patience):
                    print(f'Early stopping acitvated after {epoch+1} epochs')
                    break
    
    # finish
    train_time = time.time() - start_time
    model.load_state_dict(best_model)
    wandb.log({"train_time": train_time})
    wandb.finish()

def extract_features(extractor:nn.Module, dataloader:DataLoader):
    features = []
    labels = []
    extractor.to(device).eval()
    with torch.no_grad():
        for (xs, ys) in dataloader:
            xs = xs.to(device)
            outputs = extractor(xs)
            outputs = outputs.flatten(start_dim=1)
            features.append(outputs.cpu())
            labels.append(ys)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()

def extract_and_classical(model:nn.Module, train_set, test_set):
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    dl_train = get_dataloader(train_set, 64)
    dl_test = get_dataloader(test_set, 64)
    train_features, train_labels = extract_features(feature_extractor, dl_train)
    test_features, test_labels = extract_features(feature_extractor, dl_test)

    svm = LinearSVC()
    svm.fit(train_features, train_labels)
    y_pred = svm.predict(test_features)
    accuracy = accuracy_score(test_labels, y_pred)
    print(f'Accuracy SVM with: {accuracy:.4f}')

    logreg = LogisticRegression(max_iter=500)
    logreg.fit(train_features, train_labels)
    y_pred = logreg.predict(test_features)
    accuracy = accuracy_score(test_labels, y_pred)
    print(f'Accuracy Logistic Regression: {accuracy:.4f}')