import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GraphConv,global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import BatchNorm1d
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, matthews_corrcoef, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from datetime import datetime
import tqdm

num_gpus = torch.cuda.device_count()
currentDeviceIndex = torch.cuda.current_device()
default_devide = torch.cuda.get_device_name(currentDeviceIndex)
deviceToUse = torch.device(f'cuda:{currentDeviceIndex}' if torch.cuda.is_available() else 'cpu')
print(f'default devide:{currentDeviceIndex} {default_devide}')
if num_gpus > 0:
    print(f"{num_gpus} Physical GPUs available")
    
    # Print detailed information about each GPU
    for i in range(num_gpus):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available")
    
class BaseGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5, **kwargs):
        super(BaseGNN,self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(self._get_conv_layer(input_dim, hidden_dim, **kwargs))
        self.bns.append(BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(self._get_conv_layer(hidden_dim, hidden_dim, **kwargs))
            self.bns.append(BatchNorm1d(hidden_dim))

        # Output layer
        self.convs.append(self._get_conv_layer(hidden_dim, hidden_dim, **kwargs))
        self.lin = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    # def _get_conv_layer(self, in_channels, out_channels, **kwargs):
    #     raise NotImplementedError("Subclasses must implement this method")

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.leaky_relu(x)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x,batch)  # Global mean pooling
        x = self.lin(x)
        return torch.sigmoid(x)

class GraphSAGE(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5, **kwargs):
        super(GraphSAGE,self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, **kwargs)

    def _get_conv_layer(self, in_channels, out_channels, **kwargs):
        return SAGEConv(in_channels, out_channels, **kwargs)

class GCN(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5, **kwargs):
        super(GCN,self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, **kwargs)

    def _get_conv_layer(self, in_channels, out_channels, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)

class GAT(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5, **kwargs):
        super(GAT, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, **kwargs)

    def _get_conv_layer(self, in_channels, out_channels, **kwargs):
        return GATConv(in_channels, out_channels, **kwargs)

class GraphConvImp(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5, **kwargs):
        super(GraphConvImp, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, **kwargs)

    def _get_conv_layer(self, in_channels, out_channels, **kwargs):
        return GraphConv(in_channels, out_channels, **kwargs)

class GNNModel:
    def __init__(self, model_type, input_dim=3, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.5, 
                 epoch=100, patience=30, regularization='l2', weight_decay=1e-4, lr=0.01, **model_kwargs):
        model = None
        if model_type == 'GraphSAGE':
            model = GraphSAGE(input_dim, hidden_dim, output_dim, num_layers, dropout, **model_kwargs)
            
        elif model_type == 'GCN':
            model = GCN(input_dim, hidden_dim, output_dim, num_layers, dropout, **model_kwargs)
            
        elif model_type == 'GAT':
            model = GAT(input_dim, hidden_dim, output_dim, num_layers, dropout, **model_kwargs)
            
        elif model_type == 'GraphConv':
            model = GraphConvImp(input_dim, hidden_dim, output_dim, num_layers, dropout, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model = model.to(deviceToUse)

        print(self.model.convs, flush=True)
        self.regularization = regularization
        self.weight_decay = weight_decay
        self.optimizer = self._get_optimizer(lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epoch, eta_min=1e-6)
        self.criterion = torch.nn.BCELoss()
        self.epoch = epoch
        self.patience = patience
        self.train_losses = []
        self.val_losses = []
        self.validation_freq = 500
        self.model_type = model_type

    def create_log_directory(self, model_type, def_log_dir = ''):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/{model_type}_{timestamp}"
        if def_log_dir != '':
            log_dir = f'{def_log_dir}/{log_dir}'
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _get_optimizer(self, lr):
        if self.regularization == 'l2':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.regularization == 'l1':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported regularization type: {self.regularization}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, plot_dir=''):
        self.model.train()
        total_loss = 0
        batch_count = 0
        for data in tqdm.tqdm(train_loader):
            data = data.to(deviceToUse)
            self.optimizer.zero_grad()
            data_x = data.x
            data_y = data.y
            data_edge_index = data.edge_index
            data_batch = data.batch
            out = self.model(data_x, data_edge_index,data_batch)

            loss = self.criterion(out.view(-1), data_y.float()) 

            if self.regularization == 'l1':
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss += self.weight_decay * l1_norm

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            batch_count += 1

            if val_loader and batch_count % self.validation_freq == 0:
                val_loss, val_metrics = self.validate(val_loader,plot_dir)
                print(f"[Batch {batch_count}] Validation Loss: {val_loss:.4f}, Metrics: {val_metrics}")
        
        self.scheduler.step()
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)

        if val_loader:
            val_loss, _ = self.validate(val_loader, plot_dir)
        return avg_loss
    
    
    def validate(self, val_loader: DataLoader, log_plot_dir=''):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in tqdm.tqdm(val_loader): 
                data.to(deviceToUse)
                data_x = data.x
                data_y = data.y
                data_edge_index = data.edge_index
                data_batch = data.batch
                out = self.model(data_x, data_edge_index,data_batch)
                loss = self.criterion(out.view(-1), data_y.float())
                total_loss += loss.item()
                pred = (out > 0.5).float().squeeze()
                all_preds.append(pred.cpu())
                all_labels.append(data_y.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        avg_loss = total_loss / len(val_loader)
    
        self.val_losses.append(avg_loss)
        
        metrics = self.calculate_metrics(all_labels, all_preds)
        self._log_metrics(avg_loss, metrics,log_plot_dir)
        return avg_loss, metrics


    def log_losses(self,epoch,train_loss, val_loss=None, log_plot_dir=''):
        loss_file = os.path.join(log_plot_dir, "losses.csv")
        with open(loss_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # File is empty, write header
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
            if val_loss:
                writer.writerow([epoch, train_loss, val_loss])
            else:
                writer.writerow([epoch, train_loss,'N/A'])

    def _log_metrics(self, avg_loss, metrics, log_plot_dir=''):
        metrics_file = os.path.join(log_plot_dir, "metrics.csv")
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # File is empty, write header
                writer.writerow(['Epoch', 'Avg Loss', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'AUC ROC', 'MCC','confusion_matrix'])
            writer.writerow([len(self.train_losses)-1, avg_loss, metrics['accuracy'], metrics['f1'], metrics['precision'], 
                            metrics['recall'], metrics['sensitivity'], metrics['specificity'], metrics['auc_roc'], metrics['mcc'], metrics['confusion_matrix']])

    def calculate_metrics(self, true_labels, predictions):
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc_roc = roc_auc_score(true_labels, predictions)
        mcc = matthews_corrcoef(true_labels, predictions)

        return {
            'true_Labels': true_labels,
            'predictions': predictions,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc_roc': auc_roc,
            'mcc': mcc,
            'confusion_matrix': cm
        }
        

    def plot_losses(self, train_losses, val_losses, plot_dir='new_loss_plots'):
        print(f"Train Losses: {train_losses}", flush=True)
        print(f"Validation Losses: {val_losses}", flush=True)

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', linestyle='--', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_type} - Training and Validation Loss')
        plt.legend()
        
        plot_dir = os.path.join(plot_dir, 'loss_plots')
        # Create directory if it doesn't exist
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir, f'loss_plot_{self.model_type}_{self.regularization}.png'))
        plt.close()


    def plot_confusion_matrix(self, true_labels,predictions, plot_dir='gnn_confusion_matrix'):
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{self.model_type} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plot_dir = os.path.join(plot_dir, 'confusion_matrix')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, f'confusion_matrix_{self.model_type}_{self.regularization}.png'))
        plt.close()

    def plot_roc_curve(self, true_labels, predictions, plot_dir='gnn_roc_curve'):
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_type} - Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        plot_dir = os.path.join(plot_dir, 'roc_curve')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, f'roc_curve_{self.model_type}_{self.regularization}.png'))
        plt.close()

