"""
Checkpoint Manager for VAE training with Hugging Face Hub integration
"""
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from huggingface_hub import upload_file, HfFileSystem, hf_hub_download
from config import REPO_ID, REPO_TYPE, HF_TOKEN, CHECKPOINT_DIR


class CheckpointManager:
    """Manages model checkpoints and metrics with Hugging Face Hub integration"""
    
    def __init__(self, repo_id: str = REPO_ID, repo_type: str = REPO_TYPE):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.checkpoint_dir = CHECKPOINT_DIR
        self.metrics_file = Path("metrics.json")
        
        # Initialize metrics tracking
        self.metrics = {
            "epochs": [],
            "train_rec_loss": [],
            "train_kl_loss": [],
            "total_loss": [],
            "learning_rate": []
        }
    
    def save_checkpoint(self, 
                       model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       filename: Optional[str] = None) -> str:
        """Save checkpoint locally and optionally to Hub"""
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Use tqdm.write for better integration with progress bars
        try:
            from tqdm import tqdm
            write_func = tqdm.write
        except ImportError:
            write_func = print
        
        write_func(f"💾 [CHECKPOINT] Saving epoch {epoch:02d}...")
        
        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": {
                "batch_size": 128,
                "z_size": 512,
                "layer_count": 5,
                "im_size": 128
            }
        }
        
        # Save locally
        torch.save(checkpoint, checkpoint_path)
        write_func(f"   📁 Local: {checkpoint_path}")
        
        # Update metrics
        self._update_metrics(epoch, metrics)
        
        # Save metrics locally
        self._save_metrics()
        
        # Upload to Hub if token is available
        if HF_TOKEN:
            try:
                write_func(f"   ☁️  Uploading to Hub: {self.repo_id}")
                self._upload_to_hub(checkpoint_path, filename)
                self._upload_metrics_to_hub()
                write_func(f"   ✅ Hub upload complete")
            except Exception as e:
                write_func(f"   ❌ Hub upload failed: {e}")
        else:
            write_func(f"   ⚠️  Skipping Hub upload (no token)")
        
        return str(checkpoint_path)
    
    def load_latest_checkpoint(self, 
                              model: torch.nn.Module, 
                              optimizer: torch.optim.Optimizer) -> Tuple[int, Dict[str, float]]:
        """Load the latest checkpoint from local storage or Hub"""
        
        # Use tqdm.write for better integration with progress bars
        try:
            from tqdm import tqdm
            write_func = tqdm.write
        except ImportError:
            write_func = print
        
        write_func("🔍 [CHECKPOINT] Looking for existing checkpoints...")
        
        # Try local first
        local_checkpoint = self._get_latest_local_checkpoint()
        if local_checkpoint:
            write_func(f"   📁 Found local checkpoint: {local_checkpoint.name}")
            return self._load_checkpoint(local_checkpoint, model, optimizer)
        
        # Try Hub if no local checkpoint
        if HF_TOKEN:
            try:
                write_func(f"   ☁️  Checking Hub: {self.repo_id}")
                hub_checkpoint = self._download_latest_from_hub()
                if hub_checkpoint:
                    write_func(f"   ✅ Downloaded from Hub: {hub_checkpoint}")
                    return self._load_checkpoint(hub_checkpoint, model, optimizer)
                else:
                    write_func(f"   ℹ️  No checkpoints found on Hub")
            except Exception as e:
                write_func(f"   ❌ Hub check failed: {e}")
        else:
            write_func(f"   ⚠️  Skipping Hub check (no token)")
        
        write_func("   🚀 No checkpoints found. Starting from scratch.")
        return 0, {}
    
    def _get_latest_local_checkpoint(self) -> Optional[Path]:
        """Get the latest local checkpoint file"""
        if not self.checkpoint_dir.exists():
            return None
        
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            return None
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoint_files[0]
    
    def _download_latest_from_hub(self) -> Optional[str]:
        """Download the latest checkpoint from Hub"""
        try:
            fs = HfFileSystem()
            entries = fs.ls(f"{self.repo_id}/checkpoints")
            names = [e['name'] if isinstance(e, dict) else e for e in entries]
            files = sorted([n for n in names if n.endswith(".pt")])
            
            if not files:
                return None
            
            latest_path = files[-1]
            latest_name = latest_path.rsplit("/", 1)[-1]
            
            local_path = hf_hub_download(
                repo_id=self.repo_id, 
                filename=f"checkpoints/{latest_name}", 
                repo_type=self.repo_type
            )
            
            print(f"Downloaded checkpoint from Hub: {latest_name}")
            return local_path
            
        except Exception as e:
            print(f"Failed to download from Hub: {e}")
            return None
    
    def _load_checkpoint(self, 
                         checkpoint_path: Path, 
                         model: torch.nn.Module, 
                         optimizer: torch.optim.Optimizer) -> Tuple[int, Dict[str, float]]:
        """Load checkpoint into model and optimizer"""
        
        # Use tqdm.write for better integration with progress bars
        try:
            from tqdm import tqdm
            write_func = tqdm.write
        except ImportError:
            write_func = print
        
        write_func(f"   📥 Loading checkpoint data...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        write_func(f"   🔄 Loading model weights...")
        model.load_state_dict(checkpoint["model_state_dict"])
        
        write_func(f"   🔄 Loading optimizer state...")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Get epoch and metrics
        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})
        
        write_func(f"   ✅ Checkpoint loaded successfully!")
        write_func(f"   📊 Epoch: {epoch:02d}")
        if metrics:
            rec_loss = metrics.get("rec_loss", 0.0)
            kl_loss = metrics.get("kl_loss", 0.0)
            write_func(f"   📈 Previous metrics - Rec: {rec_loss:.6f}, KL: {kl_loss:.6f}")
        
        return epoch, metrics
    
    def _update_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Update metrics tracking"""
        self.metrics["epochs"].append(epoch)
        self.metrics["train_rec_loss"].append(metrics.get("rec_loss", 0.0))
        self.metrics["train_kl_loss"].append(metrics.get("kl_loss", 0.0))
        self.metrics["total_loss"].append(metrics.get("total_loss", 0.0))
        self.metrics["learning_rate"].append(metrics.get("lr", 0.0))
    
    def _save_metrics(self):
        """Save metrics to local file"""
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def _upload_to_hub(self, file_path: Path, filename: str):
        """Upload checkpoint to Hugging Face Hub"""
        upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"checkpoints/{filename}",
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message=f"Add checkpoint {filename}"
        )
        print(f"Uploaded {filename} to Hub")
    
    def _upload_metrics_to_hub(self):
        """Upload metrics to Hugging Face Hub"""
        upload_file(
            path_or_fileobj=str(self.metrics_file),
            path_in_repo="metrics.json",
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message="Update training metrics"
        )
        print("Uploaded metrics.json to Hub")
    
    def get_resume_info(self) -> Tuple[bool, int]:
        """Check if resuming is possible and return start epoch"""
        # Use tqdm.write for better integration with progress bars
        try:
            from tqdm import tqdm
            write_func = tqdm.write
        except ImportError:
            write_func = print
        
        local_checkpoint = self._get_latest_local_checkpoint()
        if local_checkpoint:
            checkpoint = torch.load(local_checkpoint, map_location="cpu")
            epoch = checkpoint.get("epoch", 0)
            write_func(f"🔄 [RESUME] Found checkpoint from epoch {epoch:02d} (local)")
            write_func(f"   🚀 Will resume training from epoch {epoch + 1:02d}")
            return True, epoch + 1
        
        if HF_TOKEN:
            try:
                hub_checkpoint = self._download_latest_from_hub()
                if hub_checkpoint:
                    checkpoint = torch.load(hub_checkpoint, map_location="cpu")
                    epoch = checkpoint.get("epoch", 0)
                    write_func(f"🔄 [RESUME] Found checkpoint from epoch {epoch:02d} (Hub)")
                    write_func(f"   🚀 Will resume training from epoch {epoch + 1:02d}")
                    return True, epoch + 1
            except:
                pass
        
        write_func("🔄 [RESUME] No checkpoints found - starting fresh")
        return False, 0
