# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import torch.utils.data
from PIL import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
import torch
from tqdm import tqdm
from config import *
from checkpoint_manager import CheckpointManager


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

im_size = 128


class CelebADataset(Dataset):
    """Modern PyTorch Dataset for CelebA images"""
    
    def __init__(self, data_list, im_size=128, normalize="0_1"):
        self.data_list = data_list
        self.im_size = im_size
        self.normalize = normalize
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        x = self.data_list[idx]
        
        # Ensure HxWxC
        if x.ndim == 2:  # grayscale
            x = np.stack([x, x, x], axis=-1)
        elif x.ndim == 3 and x.shape[2] == 4:  # RGBA -> RGB
            x = x[:, :, :3]

        # Resize with Pillow
        x = np.array(Image.fromarray(x).resize((self.im_size, self.im_size), Image.Resampling.BILINEAR))

        # To CHW
        x = x.transpose(2, 0, 1)  # (C,H,W)
        
        # Convert to float32 tensor
        x = torch.from_numpy(x.astype(np.float32))
        
        # Normalize
        if self.normalize == "0_1":
            x /= 255.0
        elif self.normalize == "m1_1":
            x = (x / 127.5) - 1.0
        
        # Return CPU tensor - DataLoader will handle device transfer
        return x


def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x)**2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


# process_batch function removed - replaced by CelebADataset.__getitem__

def main():
    # Print configuration status once
    from config import print_config_status
    print_config_status()
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Check if we can resume from checkpoint
    can_resume, start_epoch = checkpoint_manager.get_resume_info()
    
    # Initialize model
    vae = VAE(zsize=Z_SIZE, layer_count=LAYER_COUNT)
    vae.to(device)
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    # Initialize optimizer
    vae_optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999), weight_decay=1e-5)
 
    # Load checkpoint if available
    if can_resume:
        print(f"🚀 [TRAINING] Resuming from epoch {start_epoch:02d}")
        loaded_epoch, loaded_metrics = checkpoint_manager.load_latest_checkpoint(vae, vae_optimizer)
        start_epoch = loaded_epoch
    else:
        start_epoch = 0
        print("🚀 [TRAINING] Starting fresh training")

    # Create sample tensor for generation
    sample1 = torch.randn(128, Z_SIZE, device=device).view(-1, Z_SIZE, 1, 1)

    # Save initial checkpoint (epoch 0) if starting from scratch
    if start_epoch == 0:
        print("🎯 [TRAINING] Creating initial checkpoint (epoch 00)")
        initial_metrics = {
            "rec_loss": 0.0,
            "kl_loss": 0.0,
            "total_loss": 0.0,
            "lr": vae_optimizer.param_groups[0]['lr']
        }
        checkpoint_manager.save_checkpoint(vae, vae_optimizer, 0, initial_metrics)

    for epoch in range(start_epoch, TRAIN_EPOCHS):
        vae.train()

        with open('data_fold_%d.pkl' % (epoch % 5), 'rb') as pkl:
            data_train = pickle.load(pkl)

        print("Train set size:", len(data_train))

        # Create modern PyTorch Dataset and DataLoader
        dataset = CelebADataset(data_train, im_size=IM_SIZE, normalize="0_1")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        rec_loss = 0
        kl_loss = 0
        epoch_rec_loss = 0  # Track total epoch loss
        epoch_kl_loss = 0   # Track total epoch loss

        epoch_start_time = time.time()

        if (epoch + 1) % 8 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS}")
        for x in pbar:
            vae.train()
            x = x.to(device)
            # vae.zero_grad() # not needed for modern PyTorch
            vae_optimizer.zero_grad()
            # x.zero_grad() # removed - not needed for input tensors
            x.requires_grad = False
            
            rec, mu, logvar = vae(x)

            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()
            epoch_rec_loss += loss_re.item()  # Accumulate epoch total
            epoch_kl_loss += loss_kl.item()   # Accumulate epoch total
            
            # Update progress bar with current losses
            pbar.set_postfix({
                'rec_loss': f'{loss_re.item():.6f}',
                'kl_loss': f'{loss_kl.item():.6f}'
            })

            #############################################

            os.makedirs('results_rec', exist_ok=True)
            os.makedirs('results_gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 60
            i += 1
            if i % m == 0:
                rec_loss /= m
                kl_loss /= m
                tqdm.write('[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), TRAIN_EPOCHS, per_epoch_ptime, rec_loss, kl_loss))
                rec_loss = 0
                kl_loss = 0
                with torch.no_grad():
                    vae.eval()
                    x_rec, _, _ = vae(x)
                    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
                    x_rec = vae.decode(sample1)
                    resultsample = x_rec * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')

        del dataloader
        del data_train
        
        # Save checkpoint after every epoch
        epoch_metrics = {
            "rec_loss": epoch_rec_loss / (i + 1) if i > 0 else 0.0,
            "kl_loss": epoch_kl_loss / (i + 1) if i > 0 else 0.0,
            "total_loss": (epoch_rec_loss + epoch_kl_loss) / (i + 1) if i > 0 else 0.0,
            "lr": vae_optimizer.param_groups[0]['lr']
        }
        print(f"🎯 [TRAINING] Epoch {epoch + 1:02d} complete - saving checkpoint")
        checkpoint_manager.save_checkpoint(vae, vae_optimizer, epoch, epoch_metrics)
    
    print("Training finish!... save final model")
    torch.save(vae.state_dict(), "VAEmodel.pkl")

if __name__ == '__main__':
    main()
