import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

#VAE network configuration    
class BiomechPriorVAE(nn.Module):
    def __init__(self, num_dofs, latent_dim=32, hidden_dim=512):
        super(BiomechPriorVAE, self).__init__()

        self.num_dofs = num_dofs
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_dofs),
            nn.Linear(num_dofs, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_dofs)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z
    
    def decode(self, x):

        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rec_x = self.decode(z)

        return rec_x, mu, logvar

class VAEModelWrapper:
    def __init__(self, model_path, scaler_path, num_dofs=33, latent_dim=20, hidden_dim=512, device='cpu'):
        self.device = device

        self.model = BiomechPriorVAE(
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(device)
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Successfully loaded VAE model from: {model_path}")

        self.scaler = None
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Successfully loaded scaler from: {scaler_path}")

    #Process data using scaler
    def _preprocess(self, joint_angles):
        if self.scaler is not None:
            joint_angles_scaled = self.scaler.transform(joint_angles.reshape(1, -1)).flatten()
            return joint_angles_scaled
        
    def _postprocess(self, joint_angles):
        if self.scaler is not None:
            joint_angles_unscaled = self.scaler.inverse_transform(joint_angles.reshape(1, -1)).flatten()
            return joint_angles_unscaled

    #Reconstruct the joint angle (without gradient)
    def reconstruct(self, joint_angles):
        processed_angles = self._preprocess(joint_angles)

        with torch.no_grad():
            x = torch.tensor(processed_angles, dtype=torch.float32, device=self.device).unsqueeze(0)
            rec_x, mu, logvar = self.model.forward(x)
            rec_x_np = rec_x.squeeze(0).cpu().numpy()
            output = self._postprocess(rec_x_np)

            return output

    #Reconstruct the joint angle (with gradient w.r.t original scale)
    def reconstruct_withgrad(self, joint_angles):
            processed_angles = self._preprocess(joint_angles)
            x = torch.tensor(processed_angles, dtype=torch.float32, requires_grad=True, device=self.device)
            if x.grad is not None:
                x.grad.zero_()

            x_batch = x.unsqueeze(0)
            rec_x_batch, mu, logvar = self.model.forward(x_batch)
            rec_x = rec_x_batch.squeeze(0)
            recon_loss = F.mse_loss(x, rec_x, reduction='sum')
            recon_loss.backward()
            #Get gradient (the x.grad here refers to the gradient w.r.t the scaled input, so we also need to get the gradient w.r.t the original input (according to the chain rule))
            gradient = x.grad.detach().cpu().numpy()
            if self.scaler is not None:
                gradient = gradient / self.scaler.scale_

            rec_x_np = rec_x.detach().cpu().numpy()
            output = self._postprocess(rec_x_np)

            return output, gradient

#Initialize a instance to get the result
_vae_instance = None

def initialize_vae(model_path, scaler_path, num_dofs=33, latent_dim=20, hidden_dim=512, device='cpu'):
    global _vae_instance
    try:
        _vae_instance = VAEModelWrapper(
            model_path=model_path,
            scaler_path=scaler_path,
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        return True
    except Exception as e:
        print(f"Failed to initialize VAE: {e}")
        return False

def reconstruct(joint_angles_np):
    global _vae_instance
    if _vae_instance is None:
        raise RuntimeError("VAE instances is not initialized, call initialize_vae first!")
    
    joint_angles = joint_angles_np.flatten()
    recon_angles = _vae_instance.reconstruct(joint_angles)

    return recon_angles

def reconstruct_withgrad(joint_angles_np):
    global _vae_instance
    if _vae_instance is None:
        raise RuntimeError("VAE instances is not initialized, call initialize_vae first!")
    
    joint_angles = joint_angles_np.flatten()
    recon_angles, gradient = _vae_instance.reconstruct_withgrad(joint_angles)

    return recon_angles, gradient
