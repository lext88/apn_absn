"""If the attention mechanisms from nlsa.py, lsa.py, cca.py, and se.py already meet your requirements for handling textual data effectively, 
you may not need to integrate sce.py entirely. However, integrating sce.py might provide additional context-aware features that could enhance 
your model's performance, especially if you find its approach to self-attention and feature normalization to be beneficial."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Generate context-aware descriptor using self-attention
def generate_contextual_descriptor(data):
    b, c, h, w = data.shape
    data_flat = data.view(b, c, -1)  # (b, c, h*w)
    
    attention_scores = torch.bmm(data_flat.transpose(1, 2), data_flat)  # (b, h*w, h*w)
    attention_scores = F.softmax(attention_scores, dim=-1)  # Softmax to get attention weights
    
    contextual_features = torch.bmm(attention_scores, data_flat.transpose(1, 2)).view(b, c, h, w)  # (b, c, h, w)
    
    return contextual_features

# Normalizing the feature maps
def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)

# Multi-Head Self-Attention Layer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, embed_size = x.shape
        
        values = self.values(x).view(N, seq_length, self.heads, self.head_dim)
        keys = self.keys(x).view(N, seq_length, self.heads, self.head_dim)
        queries = self.queries(x).view(N, seq_length, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)  # (N, heads, seq_length, head_dim)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])  # (N, heads, seq_length, seq_length)
        attention = F.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)  # (N, heads, seq_length, seq_length)

        out = torch.einsum("nhql,nhld->nhqd", [attention, values]).reshape(N, seq_length, self.heads * self.head_dim)  # (N, seq_length, embed_size)
        return self.fc_out(out)  # Final linear layer

# Enhanced Context-Aware Encoder
class EnhancedContextAwareEncoder(nn.Module):
    '''
    Enhanced Context-Aware Feature Encoder for NLP tasks.
    Input:
        x: feature of shape (b,c,h,w)
    Output:
        feature_embd: context-aware semantic feature of shape (b, c + c, h, w)
    '''

    def __init__(self, planes=None, embed_size=256, heads=8):
        super(EnhancedContextAwareEncoder, self).__init__()
        self.conv1x1_in = nn.Sequential(
            nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[1]),
            nn.ReLU(inplace=True)
        )
        
        self.embeddingFea = nn.Sequential(
            nn.Conv2d(planes[1] + planes[1], planes[2], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[2]),
            nn.ReLU(inplace=True)
        )
        
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[2], planes[3], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[3])
        )
        
        self.self_attention = MultiHeadSelfAttention(embed_size, heads)

    def forward(self, x):
        # Channel reduction
        x = self.conv1x1_in(x)

        # Generate context-aware features
        feature_gs = generate_contextual_descriptor(x)

        # Concatenate original features with contextual features
        feature_cat = torch.cat([x, feature_gs], 1)

        # Embed concatenated features
        feature_embd = self.embeddingFea(feature_cat)

        # Reshape for attention: (b, c, h, w) to (b, h*w, c)
        b, c, h, w = feature_embd.shape
        feature_embd_flat = feature_embd.view(b, c, -1).transpose(1, 2)  # (b, h*w, c)

        # Apply multi-head self-attention
        attention_output = self.self_attention(feature_embd_flat)

        # Reshape back to original dimensions
        attention_output = attention_output.transpose(1, 2).view(b, c, h, w)

        # Channel expansion
        feature_embd = self.conv1x1_out(attention_output)
        
        return feature_embd
