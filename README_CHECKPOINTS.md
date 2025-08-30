# VAE Training with Checkpointing and Hugging Face Hub Integration

This project now includes a robust checkpointing system that automatically saves your training progress and can resume from where you left off.

## 🚀 Features

- **Automatic Checkpointing**: Saves model state every 5 epochs
- **Resume Training**: Automatically detects and loads the latest checkpoint
- **Hugging Face Hub Integration**: Uploads checkpoints and metrics to your Hub repository
- **Local + Cloud Backup**: Checkpoints saved both locally and on Hub
- **Metrics Tracking**: Automatic logging of training metrics for Hub charts

## 📁 File Structure

```
VAE/
├── config.yaml              # Your configuration (edit this with your token)
├── config.template.yaml     # Template (safe to commit)
├── config.py                # Configuration loader
├── checkpoint_manager.py    # Checkpoint management and Hub integration
├── VAE.py                  # Main training script (updated)
├── .gitignore              # Prevents sensitive files from being committed
└── checkpoints/            # Local checkpoint storage
```

## 🔐 Setup

### 1. Create Hugging Face Repository

1. Go to [Hugging Face Hub](https://huggingface.co/)
2. Create a new model repository (e.g., `your-username/vae-celeba`)

### 2. Get Your Hugging Face Token

1. Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "write" permissions
3. Copy the token

### 3. Set Up Configuration (Windows-Friendly!)

**Option A: Edit config.yaml directly (Recommended)**
```yaml
# In config.yaml, edit these fields:
huggingface:
  username: "your_actual_username"  # Your HF username
  repo_name: "vae-celeba"           # Your repo name
  token: "your_actual_token_here"   # Your HF token
```

**Option B: Copy from template**
```cmd
# Windows Command Prompt
copy config.template.yaml config.yaml

# Then edit config.yaml with your values
notepad config.yaml
```

**Option C: PowerShell**
```powershell
# PowerShell
Copy-Item config.template.yaml config.yaml

# Then edit config.yaml with your values
notepad config.yaml
```

## 🎯 Usage

### Start Training
```bash
uv run VAE.py
```

The script will automatically:
- Check for existing checkpoints
- Resume from the latest checkpoint if available
- Save checkpoints every 5 epochs
- Upload checkpoints and metrics to Hub

### Resume Training
If training is interrupted, simply run the same command:
```bash
uv run VAE.py
```

The script will automatically detect and load the latest checkpoint from either:
1. Local storage (`checkpoints/` directory)
2. Hugging Face Hub (if local not available)

## 📊 Metrics and Visualization

### Local Metrics
- Metrics are saved to `metrics.json`
- Includes epoch-by-epoch loss tracking
- Learning rate changes

### Hub Integration
- Metrics automatically uploaded to Hub
- View training curves on your model page
- Checkpoints available for download

## 🔧 Configuration

Edit `config.py` to customize:
- Checkpoint frequency (`CHECKPOINT_INTERVAL`)
- Sample generation frequency (`SAMPLE_INTERVAL`)
- Model parameters (batch size, z-size, etc.)
- Repository settings

## 🛡️ Security

- **`.env` file is gitignored** - your token won't be committed
- **Checkpoints are gitignored** - large files won't bloat your repo
- **Environment variables** - can be set securely in production

## 📝 Example config.yaml File

```yaml
# config.yaml (DO NOT commit this file)
huggingface:
  username: "AssafR2"
  repo_name: "vae-celeba"
  repo_type: "model"
  token: "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Your actual token

training:
  batch_size: 128
  z_size: 512
  epochs: 40
  learning_rate: 0.0005
  checkpoint_interval: 5
  sample_interval: 60

model:
  layer_count: 5
  im_size: 128

paths:
  checkpoint_dir: "checkpoints"
  results_dir: "results"
  data_dir: "."
```

## 🚨 Troubleshooting

### "Hugging Face token not set" Warning
- Edit `config.yaml` and set your token in the `huggingface.token` field
- Or copy from template: `copy config.template.yaml config.yaml`
- Check token permissions (needs "write" access)

### Checkpoint Upload Fails
- Verify your token has write permissions
- Check repository exists and is accessible
- Ensure internet connection

### Resume Not Working
- Check `checkpoints/` directory exists
- Verify checkpoint files are valid
- Check Hub repository permissions

## 🔄 Manual Checkpoint Management

### Save Checkpoint
```python
from checkpoint_manager import CheckpointManager

manager = CheckpointManager()
manager.save_checkpoint(model, optimizer, epoch, metrics)
```

### Load Checkpoint
```python
epoch, metrics = manager.load_latest_checkpoint(model, optimizer)
```

### Check Resume Status
```python
can_resume, start_epoch = manager.get_resume_info()
```

## 📈 Benefits

1. **Never Lose Progress**: Automatic checkpointing every 5 epochs
2. **Cloud Backup**: Checkpoints stored on Hugging Face Hub
3. **Easy Resume**: Just run the script again
4. **Metrics Tracking**: Beautiful training curves on Hub
5. **Collaboration**: Share checkpoints with team members
6. **Production Ready**: Robust error handling and fallbacks

Your training is now bulletproof! 🎉
