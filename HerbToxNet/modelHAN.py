import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=True),
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(in_size, out_size, layer_num_heads, dropout, activation=F.elu)
            )
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        attention_scores = []
        for i, g in enumerate(gs):
            embedding, attention = self.gat_layers[i](g, h, get_attention=True)
            semantic_embeddings.append(embedding.flatten(1))
            attention_scores.append(attention)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings), attention_scores

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l - 1], hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, gs, h):
        for gnn in self.layers:
            h, attention_scores = gnn(gs, h)
        h = self.predict(h)
        return h, attention_scores

class MultiLabelToxicityClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, dropout_prob=0.2, output_dim=5):
        super(MultiLabelToxicityClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)

def dynamic_contrastive_loss(embeddings, labels, t=2, temperature=0.5):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    cos_sim = torch.matmul(embeddings, embeddings.T)
    labels = labels.float()
    label_sim = torch.matmul(labels, labels.T) / (labels.sum(dim=1, keepdim=True) + 1e-8)
    dynamic_coeff = label_sim ** t
    exp_sim = torch.exp(cos_sim / temperature)
    positive_sim = exp_sim * dynamic_coeff
    mask = ~torch.eye(cos_sim.size(0), dtype=torch.bool).to(cos_sim.device)
    positive_sim = positive_sim * mask
    exp_sim = exp_sim * mask
    loss = -torch.log((positive_sim.sum(dim=1) + 1e-8) / (exp_sim.sum(dim=1) + 1e-8))
    return loss.mean()

def weighted_label_fusion(test_emb, train_embs, train_labels, sigma=0.2, similarity_metric='jaccard'):
    if similarity_metric == 'cosine':
        sim = F.cosine_similarity(test_emb.unsqueeze(0), train_embs, dim=1)
    elif similarity_metric == 'euclidean':
        sim = -torch.norm(train_embs - test_emb.unsqueeze(0), dim=1)
    elif similarity_metric == 'jaccard':
        # Generalized Jaccard (Tanimoto) for continuous vectors
        # J(A, B) = (A . B) / (||A||^2 + ||B||^2 - A . B)
        dot_product = (test_emb.unsqueeze(0) * train_embs).sum(dim=1)
        norm_sq_test = (test_emb**2).sum()
        norm_sq_train = (train_embs**2).sum(dim=1)
        denominator = norm_sq_test + norm_sq_train - dot_product
        # Avoid division by zero
        sim = dot_product / (denominator + 1e-8)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
    mask = sim >= sigma
    filtered_sim = sim[mask]
    filtered_labels = train_labels[mask]
    if filtered_sim.numel() == 0:
        return torch.zeros(train_labels.size(1)).to(train_labels.device)
    weights = F.softmax(filtered_sim, dim=0)
    fused_label = torch.matmul(weights, filtered_labels)
    return fused_label

class GTConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A):
        A = torch.sum(A * F.softmax(self.weight, dim=1), dim=1)
        return A

class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first:
            a1 = self.conv1(A)
            a2 = self.conv2(A)
            return torch.matmul(a1, a2)
        else:
            a1 = self.conv1(A)
            return torch.matmul(H_, a1)

class GTN(nn.Module):
    def __init__(self, num_edge_types, num_channels, in_dim, hidden_dim, out_dim, num_layers=2):
        super(GTN, self).__init__()
        self.num_edge_types = num_edge_types
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.is_norm = True

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GTLayer(num_edge_types, num_channels, first=True))
            else:
                self.layers.append(GTLayer(num_edge_types, num_channels, first=False))

        self.gcn = GCN(in_dim, hidden_dim, out_dim)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            if self.is_norm:
                deg = torch.sum(H[i], dim=1)
                deg[deg == 0] = 1
                deg_inv = deg.pow(-1)
                norm_H.append(deg_inv.view(-1, 1) * H[i])
            else:
                norm_H.append(H[i])
        return torch.stack(norm_H)

    def forward(self, A, X):
        # A: (num_edge_types, N, N)
        # X: (N, in_dim)
        
        # Learn meta-paths
        H = A
        for layer in self.layers:
            H = layer(A, H)
        
        # H is now (num_channels, N, N) - the learned adjacency matrices
        H = self.normalization(H)
        
        # Run GCN on learned graphs
        # We process each channel separately and then aggregate
        embeddings = []
        for i in range(self.num_channels):
            embeddings.append(self.gcn(X, H[i]))
            
        # Aggregate (e.g., concat or mean)
        # HAN concatenates if multi-head, here we have channels.
        # Let's concatenate and then project if needed, or just sum.
        # Original GTN concatenates.
        embeddings = torch.cat(embeddings, dim=1) 
        return embeddings

class FastGTConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FastGTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # FastGTN: use 1x1 conv weight to mix adjacency matrices efficiently
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A):
        # A: (in_channels, N, N)
        # Weight: (out_channels, in_channels, 1, 1)
        # We want to produce (out_channels, N, N)
        # This is essentially a 1x1 convolution on the 'channels' dimension of A
        
        # Optimization: Use torch.sum instead of full conv2d overhead if possible,
        # or just broadcast multiply.
        # A is (C_in, N, N)
        # W is (C_out, C_in, 1, 1)
        # Output_i = Sum_j (W_ij * A_j)
        
        # Using softmax on weights as per GTN paper for stability
        w = F.softmax(self.weight, dim=1)
        
        # (C_out, C_in, 1, 1) * (1, C_in, N, N) -> (C_out, C_in, N, N) -> sum(1) -> (C_out, N, N)
        A_out = torch.sum(w * A.unsqueeze(0), dim=1)
        return A_out

class FastGTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first=True):
        super(FastGTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first:
            self.conv1 = FastGTConv(in_channels, out_channels)
            self.conv2 = FastGTConv(in_channels, out_channels)
        else:
            self.conv1 = FastGTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first:
            a1 = self.conv1(A)
            a2 = self.conv2(A)
            # Matrix multiplication of two learned adjacency sets
            # a1: (out_channels, N, N)
            # a2: (out_channels, N, N)
            # Result: (out_channels, N, N)
            return torch.matmul(a1, a2)
        else:
            a1 = self.conv1(A)
            return torch.matmul(H_, a1)

class FastGTN(nn.Module):
    def __init__(self, num_edge_types, num_channels, in_dim, hidden_dim, out_dim, num_layers=2):
        super(FastGTN, self).__init__()
        self.num_edge_types = num_edge_types
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.is_norm = True

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(FastGTLayer(num_edge_types, num_channels, first=True))
            else:
                self.layers.append(FastGTLayer(num_edge_types, num_channels, first=False))

        self.gcn = GCN(in_dim, hidden_dim, out_dim)

    def normalization(self, H):
        # Optimized normalization
        if not self.is_norm:
            return H
            
        # H: (channels, N, N)
        # Calculate degree matrix D^-1
        deg = torch.sum(H, dim=2, keepdim=True) # (channels, N, 1)
        deg[deg == 0] = 1
        deg_inv = deg.pow(-1)
        
        # Row normalization: D^-1 * H
        H_norm = deg_inv * H
        return H_norm

    def forward(self, A, X):
        # A: (num_edge_types, N, N)
        # X: (N, in_dim)
        
        H = A
        for layer in self.layers:
            H = layer(A, H)
        
        H = self.normalization(H)
        
        # Optimized GCN step:
        # Instead of looping through channels, use batch matrix multiplication
        # X: (N, D)
        # H: (C, N, N)
        # We want: List of GCN(X, H[i])
        
        # 1. Project X first (shared weights for first GCN layer usually)
        # In our GCN class, weights are shared? Let's check GCN implementation.
        # Yes, GCNConv has one weight matrix.
        
        # Pre-compute XW: (N, hidden)
        X_hidden = self.gcn.layer1.forward_base(X) 
        
        # 2. Multiply with Adjacency matrices (C, N, N) @ (1, N, hidden) -> (C, N, hidden)
        # H: (C, N, N)
        # X_hidden: (N, H)
        # We can use matmul with broadcasting
        H_emb = torch.matmul(H, X_hidden) # (C, N, hidden)
        H_emb = F.relu(H_emb)
        
        # 3. Second GCN layer
        # Pre-compute HW: (C, N, out) -> This is tricky because GCNConv weights are (hidden, out)
        # We apply linear transform to the last dimension
        W2 = self.gcn.layer2.weight # (hidden, out)
        bias2 = self.gcn.layer2.bias # (out)
        
        # (C, N, hidden) @ (hidden, out) -> (C, N, out)
        embeddings = torch.matmul(H_emb, W2)
        
        # Apply Adjacency again? Standard GCN is A(XW).
        # Here our "A" is H[i].
        # So Layer 2 is: H[i] @ (ReLU(H[i] @ X @ W1)) @ W2 + b
        
        # We already have H_emb = H[i] @ X @ W1 (activated)
        # Now we need H[i] @ H_emb @ W2
        
        # First project H_emb: (C, N, hidden) @ (hidden, out) -> (C, N, out)
        H_emb_proj = torch.matmul(H_emb, W2)
        
        # Then multiply by H: (C, N, N) @ (C, N, out) -> (C, N, out)
        final_emb = torch.matmul(H, H_emb_proj) + bias2
        final_emb = F.relu(final_emb)
        
        # Concatenate channels: (C, N, out) -> (N, C * out)
        final_emb = final_emb.permute(1, 0, 2).reshape(X.shape[0], -1)
        
        return final_emb

class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward_base(self, X):
        # Just X @ W + b (partially)
        return torch.matmul(X, self.weight)

    def forward(self, X, A):
        h = torch.matmul(X, self.weight)
        output = torch.matmul(A, h) + self.bias
        return F.relu(output)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.layer1 = GCNConv(in_dim, hidden_dim)
        self.layer2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, X, A):
        h = self.layer1(X, A)
        h = self.layer2(h, A)
        return h
