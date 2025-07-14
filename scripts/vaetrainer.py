import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import threading
import time
from tqdm import tqdm
import pickle

from b3dconverter import Gait3dB3DConverter
from datavisualize import PoseVisualizer

def load_data(data_root_path, geometry_path, batch_size=64, train_split=0.8):
    all_datafile_path = []
    file_idx = 0
    for root, dirs, files in os.walk(data_root_path):
        for file in files:
            if file.endswith('.b3d'):
                datafile_path = os.path.join(root, file)
                all_datafile_path.append(datafile_path)
                file_idx += 1

    print(f"Loaded {file_idx} data, ready for converting...")

    #Initialize data converter
    converter = Gait3dB3DConverter(geometry_path)

    all_data = []
    for datafile in all_datafile_path:
        subject = converter.load_subject(datafile, processing_pass=0)
        joint_pos = converter.convert_data(
            subject, processing_pass=0
        )
        all_data.append(joint_pos)

    combined_data = np.vstack(all_data)
    print(f"Data converting complete! Data shape: {combined_data.shape}")
    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(combined_data)
    
    dataset = TensorDataset(torch.FloatTensor(normalized_data))

    train_size = int(train_split * len(dataset))
    val_size = int(len(dataset) - train_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler


#VAE network configuration    
class BiomechPriorVAE(nn.Module):
    def __init__(self, num_dofs, latent_dim=20, hidden_dim=512):
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
    

def vae_loss(x, recon_x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


class BiomechPriorVAETrainer:
    def __init__(self,
                 model: BiomechPriorVAE,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'loss': [], 'recon_loss': [], 'kl_loss': []}

    def train_epoch(self,
                    train_loader,
                    optimizer: torch.optim.Optimizer,
                    beta=1.0):
        self.model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for batch_idx, (data,) in enumerate(tqdm(train_loader, desc="Training")):
            data = data.to(self.device)
            optimizer.zero_grad()

            recon_data, mu, logvar = self.model(data)
            loss, recon_loss, kl_loss = vae_loss(x=data, recon_x=recon_data, mu=mu, logvar=logvar, beta=beta)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

        num_batches = len(train_loader)

        return {
            'loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon_loss / num_batches,
            'kl_loss': epoch_kl_loss / num_batches
        }
        
    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, beta=1.0, save_path=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            #Train
            train_metrics = self.train_epoch(train_loader=train_loader, optimizer=optimizer, beta=beta)

            #Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader=val_loader, beta=beta)
                scheduler.step(val_metrics['loss'])

                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"Train Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Recon: {val_metrics['recon_loss']:.4f}, KL: {val_metrics['kl_loss']:.4f}")

                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        print("Best model saved!")
                
            else:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"Train Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}")

            self.training_history['loss'].append(train_metrics['loss'])
            self.training_history['recon_loss'].append(train_metrics['recon_loss'])
            self.training_history['kl_loss'].append(train_metrics['kl_loss'])

    def validate(self, val_loader, beta=1.0):
        self.model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0

        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(self.device)

                recon_data, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = vae_loss(x=data, recon_x=recon_data, mu=mu, logvar=logvar, beta=beta)

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()

            num_batches = len(val_loader)

            return{
                'loss': val_loss / num_batches,
                'recon_loss': val_recon_loss / num_batches,
                'kl_loss': val_kl_loss / num_batches
            }
    
    def test(self, val_loader, scaler):
        self.model.eval()
        result = []

        with torch.no_grad():
            for i, (data,) in enumerate(val_loader):
                data = data.to(self.device)
                recon_data, mu, logvar = self.model(data)
                
                ori_data = data.cpu().numpy()
                rec_data = recon_data.cpu().numpy()

                ori_denorm = scaler.inverse_transform(ori_data)
                rec_denorm = scaler.inverse_transform(rec_data)

                result.append({
                    'original': ori_denorm,
                    'recon': rec_denorm
                })
        
        return result

    def visualize_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(self.training_history['loss'])
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        
        axes[1].plot(self.training_history['recon_loss'])
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        
        axes[2].plot(self.training_history['kl_loss'])
        axes[2].set_title('KL Divergence Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.show()


class BioPrioVAEVisualizer:
    def __init__(self):
        self.ori_visualizer = PoseVisualizer()
        self.rec_visualizer = PoseVisualizer()

    def compare_poses(self, original_pose, recon_pose, original_port=8080, recon_port=8081):
        
        def ori_visualize():
            try:
                self.ori_visualizer.visualize_pose(
                    joint_position=original_pose,
                    port=original_port
                )
            except Exception as e:
                print(f"Error in visualize original pose: {e}")
        
        def rec_visualize():
            try:
                self.rec_visualizer.visualize_pose(
                    joint_position=recon_pose,
                    port=recon_port
                )
            except Exception as e:
                print(f"Error in visualize reconstruction pose: {e}")

        print("\n-ORIGINAL POSE: localhost:8080")
        print("-RECONSTRUCTED POSE: localhost:8081")
        print("="*50)
        ori_thread = threading.Thread(target=ori_visualize, daemon=True)
        rec_thread = threading.Thread(target=rec_visualize, daemon=True)

        ori_thread.start()
        time.sleep(1)
        rec_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
                print("Visualization stopped")


def train_model(
        data_root_path,
        geometry_path,
        output_path,
        latent_dim=20,
        batch_size=64,
        num_dofs=33,
        num_epochs=100,
        learning_rate=1e-3,
        beta=1.0,
        train_split=0.8
):
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "BiomechPriorVAE_best.pth")
    scaler_path = os.path.join(output_path, "scaler.pkl")

    train_loader, val_loader, scaler = load_data(data_root_path=data_root_path, geometry_path=geometry_path, batch_size=batch_size, train_split=train_split)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    model = BiomechPriorVAE(num_dofs=num_dofs, latent_dim=latent_dim)
    trainer = BiomechPriorVAETrainer(model=model)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        beta=beta,
        save_path=save_path
    )
    trainer.visualize_history()

    print("Training complete!")
    print("\n" + "="*50)
    print("Starting visualize test sample...")
    print("="*50)

    visualize_result(
        model=model,
        trainer=trainer,
        val_loader=val_loader,
        scaler=scaler
    )

    return model, trainer

def test_model(
        data_root_path,
        geometry_path,
        model_path,
        scaler_path,
        latent_dim=20,
        batch_size=64,
        num_dofs=33,
        train_split=0.8,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    train_loader, val_loader, _ = load_data(data_root_path=data_root_path, geometry_path=geometry_path, batch_size=batch_size, train_split=train_split)
    print("Data loaded successfully!")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully!")
    
    model = BiomechPriorVAE(num_dofs=num_dofs, latent_dim=latent_dim)
    device = 'cude' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully on {device}!")

    trainer = BiomechPriorVAETrainer(model=model, device=device)
    print("\n" + "="*50)
    print("Starting visualize test sample...")
    print("="*50)

    visualize_result(
        model=model,
        trainer=trainer,
        val_loader=val_loader,
        scaler=scaler
    )

    return model, trainer, scaler


#Visualize a sample (original&reconstructed) with skeletal model
def visualize_result(model, trainer, val_loader, scaler):
    #Random sample
    results = trainer.test(val_loader=val_loader, scaler=scaler)
    '''result shape:
    [
    #batch0
    {'original': (batch_size, num_dofs)
     'recon': (batch_size, num_dofs)
    },
    #batch1
    {'original': (batch_size, num_dofs)
     'recon': (batch_size, num_dofs)
    }
    ...
    ]
    '''
    batch_idx = np.random.randint(len(results))
    result = results[batch_idx]

    original_poses = result['original']
    recon_poses = result['recon']
    
    sample_idx = np.random.randint(len(original_poses))
    original_pose = original_poses[sample_idx]
    recon_pose = recon_poses[sample_idx]

    visualizer = BioPrioVAEVisualizer()
    visualizer.compare_poses(original_pose=original_pose, recon_pose=recon_pose)


if __name__ == "__main__":
    data_root_path = "../data/Dataset"
    output_path = "../result/model/"
    geometry_path = "../data/Geometry/"

    mode = "train" #"train" or "test"

    if mode == "train":
        print("Starting training...")
        model, trainer = train_model(
            data_root_path=data_root_path,
            geometry_path=geometry_path,
            output_path=output_path,
            latent_dim=20,
            batch_size=256,
            num_dofs=33,
            num_epochs=30,
            learning_rate=1e-3,
            train_split=0.8
        )
    
    elif mode == "test":
        print("Starting testing...")
        model_path = os.path.join(output_path, "BiomechPriorVAE_best.pth")
        scaler_path = os.path.join(output_path, "scaler.pkl")
        model, trainer, scaler = test_model(
            data_root_path=data_root_path,
            geometry_path=geometry_path,
            model_path=model_path,
            scaler_path=scaler_path,
            latent_dim=20,
            batch_size=256,
            num_dofs=33,
            train_split=0.8,
        )

    else:
        print("Invalid mode! Please select 'train' or 'test' mode.")
