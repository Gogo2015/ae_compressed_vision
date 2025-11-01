import torch
import torch.nn.functional as f
import torch.nn as nn
import math

class resblock_a(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_stack = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, 3, stride=1, padding='same'),
			torch.nn.BatchNorm3d(128),
			torch.nn.ReLU(),
            torch.nn.Conv3d(128, 128, 3, stride=1, padding='same'),
			torch.nn.BatchNorm3d(128)
        )
        
    def forward(self, x):
        residual = x
        out = self.conv_stack(x) + residual
        return out
    
class resblock_b(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resblock_a_stack = torch.nn.Sequential(
            resblock_a(),
            resblock_a(),
            resblock_a()
        )
        
    def forward(self, x):
        residual = x
        out = self.resblock_a_stack(x) + residual
        return out
    
class resblock_c(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resblock_stack = torch.nn.Sequential(
            resblock_b(),
            resblock_b(),
            resblock_b(),
            resblock_b(),
            resblock_b(),
            resblock_a()
        )
        
    def forward(self, x):
        residual = x
        out = self.resblock_stack(x) + residual
        return out
    
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        #Encoder
        self.encoderConv1 = torch.nn.Conv3d(in_channels, 64, 5, stride=(1,2,2))
        self.encoderBn1 = torch.nn.BatchNorm3d(64)
        self.encoderConv2 = torch.nn.Conv3d(64, 128, 5, stride=(1,2,2))
        self.encoderBn2 = torch.nn.BatchNorm3d(128)
        self.encoderConv3 = torch.nn.Conv3d(128, 32, 5, stride=(1,2,2))
        self.encoderBn3 = torch.nn.BatchNorm3d(32) 

        self.resblock_cEncoder = resblock_c()

    def forward(self, x, recstate):
        stride = (1,2,2)

        x = x + recstate
        #Encoder
        x = f.pad(x, self.same_pad(x, stride, 5))
        x = self.encoderConv1(x)
        x = self.encoderBn1(x)
        x = f.relu(x)
        
        x = f.pad(x, self.same_pad(x, stride, 5))
        x = self.encoderConv2(x)
        x = self.encoderBn2(x)
        x = f.relu(x)

        x = self.resblock_cEncoder(x)

        x = f.pad(x, self.same_pad(x, stride, 5))
        x = self.encoderConv3(x)
        x = self.encoderBn3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        #Decoder
        self.decoderConv1 = torch.nn.ConvTranspose3d(32, 128, 3, stride=(1,2,2), padding=1, output_padding=(0,1,1))
        self.decoderBn1 = torch.nn.BatchNorm3d(128)
        self.decoderConv2 = torch.nn.ConvTranspose3d(128, 64, 5, stride=(1,2,2), padding=2, output_padding=(0,1,1))
        self.decoderBn2 = torch.nn.BatchNorm3d(64)
        self.decoderConv3 = torch.nn.ConvTranspose3d(64, self.in_channels, 5, stride=(1,2,2), padding=2, output_padding=(0,1,1))
        self.decoderBn3 = torch.nn.BatchNorm3d(self.in_channels)

        self.resblock_cDecoder = resblock_c()

    def forward(self, x):
        #Decoder
        x = self.decoderConv1(x)
        x = self.decoderBn1(x)
        x = f.relu(x)

        x = self.resblock_cDecoder(x)

        x = self.decoderConv2(x)
        x = self.decoderBn2(x)
        x = f.relu(x)

        x = self.decoderConv3(x)
        x = self.decoderBn3(x)

        return x
        
    
class FeedbackRecurrentModule(nn.Module):
    def __init__(self, in_channels):
        super(FeedbackRecurrentModule, self).__init__()
        self.conv_gru = nn.GRUCell(input_size=in_channels, hidden_size=in_channels)
        self.decoderConv4 = torch.nn.Conv3d(self.in_channels, 32, 5, stride=1)

    def forward(self, x, prev_recstate):
        updated_recstate = self.conv_gru(x, prev_recstate)
        x = x + updated_recstate
        x = self.decoderConv4(x)
        return updated_recstate, x
    
class IPautoencoder(torch.nn.Module):
    def __init__(self, in_channels, codebook_length, device, batch_size):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        #z output from encoder as B x D x Channels x L x W
        #Initialize centroids to L x 1
        centroids = torch.ones((codebook_length,1), dtype = torch.float32, device = device)
        torch.nn.init.kaiming_uniform_(centroids, mode="fan_in", nonlinearity="relu")
        centroids = torch.squeeze(centroids)
        self.centroids = nn.Parameter(centroids)
        self.codebook_length = codebook_length

        #Layers
        self.encoder_I = Encoder(in_channels)
        self.encoder_P = Encoder(in_channels)
        self.decoder_I = Decoder(in_channels)
        self.decoder_P = Decoder(in_channels)
        self.feedbackRecurr = FeedbackRecurrentModule(in_channels)


    def forward(self, x):
        reconstructed_video = []
        for i, frame in enumerate(x):
            if i == 0: #I-Frame
                recstate = torch.zeros(size = latent.size())
                latent = self.encoder_I(frame, recstate)
                reconstructed = self.decoder_I(latent)

            else: #P-frames
                latent = self.encoder_P(frame)
                reconstructed = self.decoder_P(latent)
                recstate, reconstructed = self.feedbackRecurr(reconstructed, recstate)

            reconstructed_video.append(reconstructed)
            prev_reconstructed = reconstructed

        return reconstructed_video