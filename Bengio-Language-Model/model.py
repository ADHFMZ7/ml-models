import torch
from torch import nn

context_len = 3



class Model(nn.Module):

    def __init__(self, vocab_size, context_len):
        super(Model, self).__init__()

        # (batch, context_len)
        self.C = torch.randn((vocab_size, 2))
        
        # (batch, context_len, 2) 
        
        self.W = torch.randn((2 * context_len, 100))

        # 

    # INPUT COMES IN THE FORM: (27, CONTEXT_LENGTH, BATCH_SIZE)
    def forward(self, x, target):

        emb = self.C[x]
        logits =  self.fc(emb.view(-1, 100 * context_len))

        loss = nn.CrossEntropyLoss(logits, target)
        return logits, loss



