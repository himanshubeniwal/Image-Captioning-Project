import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # Embedding between vocab_size and embed_size
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first = True)  # Using LSTM for embed_size, hidden_size, number of layers, vocab_size)
        self.linear = nn.Linear(hidden_size, vocab_size) # Creating the Linearity for hidden_size and vocab_size
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings),1)
        
        LSTMResults, _ = self.lstm(embeddings)
        
        out = self.linear(LSTMResults[:, :-1,:])
        return out

    def sample(self, inputs, states=None):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #Empty tokens list which will be returned
        tokens = []    
        #max_length = 20
        for i in range(20):
            # Passing the inputs and states to lstm
            outputs, states= self.lstm(inputs, states) 
            
            outputs = self.linear(outputs.squeeze(1))
            
            predicted = outputs.max(1)[1]
            tokens.append(predicted.item())
            # Unsqueezing the tokens
            inputs = self.embed(predicted).unsqueeze(1)

        return tokens