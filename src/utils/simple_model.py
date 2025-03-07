import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros


class SimpleGCNLayer(MessagePassing):
    """
    A simple Graph Convolutional Network layer.
    """
    def __init__(self, in_dim, out_dim):
        super(SimpleGCNLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, edge_index):
        # x has shape [N, in_dim]
        # edge_index has shape [2, E]
        
        # Perform the message passing
        x = self.lin(x)
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        # x_j has shape [E, out_dim]
        return x_j
    
    def update(self, aggr_out):
        # aggr_out has shape [N, out_dim]
        return aggr_out


class SimpleModel(nn.Module):
    """
    A simple graph neural network model with two heads:
    1. Position denoising head
    2. Atom type prediction head
    """
    def __init__(self, node_in_dim, node_hidden_dim=256, num_layers=6, 
                 num_atom_types=5, dropout=0.2):
        super(SimpleModel, self).__init__()
        
        self.node_in_dim = node_in_dim
        self.hidden_dim = node_hidden_dim
        self.num_atom_types = num_atom_types
        self.dropout = dropout
        
        # Initial embedding layer with batch norm
        self.node_embedding = nn.Linear(node_in_dim, node_hidden_dim)
        self.bn_embed = nn.BatchNorm1d(node_hidden_dim)
        
        # GCN layers with batch norm and residual connections
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(GCNConv(node_hidden_dim, node_hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(node_hidden_dim))
        
        # Atom type prediction head
        self.atom_type_head = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim*2),
            nn.BatchNorm1d(node_hidden_dim*2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_hidden_dim*2, node_hidden_dim),
            nn.BatchNorm1d(node_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_hidden_dim, num_atom_types)
        )
        
        # Position denoising head (predicts the noise in positions)
        self.position_head = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim*2),
            nn.BatchNorm1d(node_hidden_dim*2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_hidden_dim*2, node_hidden_dim),
            nn.BatchNorm1d(node_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_hidden_dim, 3)  # 3D coordinates
        )
        
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [N, node_in_dim]
                - pos: Node positions [N, 3]
                - edge_index: Graph connectivity [2, E]
                
        Returns:
            Tuple of (atom_type_pred, position_offset_pred)
            - atom_type_pred: Predicted atom types [N, num_atom_types]
            - position_offset_pred: Predicted position noise [N, 3]
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index
        
        # Initial node embedding
        x = self.node_embedding(x)
        x = self.bn_embed(x)
        x = F.silu(x)
        
        # Apply GCN layers with residual connections and batch norm
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x_res = x  # Store for residual connection
            x = layer(x, edge_index)
            x = bn(x)
            x = F.silu(x)
            if i > 0:  # Apply residual connection after first layer
                x = x + x_res
        
        # Apply prediction heads
        atom_type_pred = self.atom_type_head(x)
        position_offset_pred = self.position_head(x)
        
        return atom_type_pred, position_offset_pred


class CombinedLoss(nn.Module):
    """
    Combined loss function for both atom type and position predictions.
    """
    def __init__(self, atom_weight=1.0, pos_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.atom_weight = atom_weight
        self.pos_weight = pos_weight
        
        # Loss functions
        self.atom_loss_fn = nn.CrossEntropyLoss()
        self.pos_loss_fn = nn.MSELoss()  # MSE is the correct loss for diffusion models
        
    def forward(self, predictions, targets):
        """
        Compute the combined loss.
        
        Args:
            predictions: Tuple of (atom_type_pred, position_offset_pred)
            targets: Tuple of (atom_type_target, position_offset_target)
            
        Returns:
            Tuple of (position_loss, atom_type_loss)
        """
        atom_type_pred, position_offset_pred = predictions
        atom_type_target, position_offset_target = targets
        
        # Make sure atom_type_target is the correct format for CrossEntropyLoss (class indices)
        if atom_type_target.dim() > 1 and atom_type_target.size(1) > 1:
            # If it's one-hot encoded, convert to class indices
            atom_type_target = torch.argmax(atom_type_target, dim=1)
        
        # Compute atom type loss (cross entropy)
        atom_loss = self.atom_loss_fn(atom_type_pred, atom_type_target)
        
        # Compute position loss (Huber/SmoothL1)
        pos_loss = self.pos_loss_fn(position_offset_pred, position_offset_target)
        
        # Return individual losses for flexible weighting in training loop
        return pos_loss, atom_loss