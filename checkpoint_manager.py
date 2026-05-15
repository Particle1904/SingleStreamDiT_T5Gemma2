import os
import torch
import glob
import re
from huggingface_hub import hf_hub_download

try:
    from huggingface_hub import HfApi
    HAS_HF = True
except ImportError:
    HAS_HF = False

class CheckpointManager:
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        
        token = getattr(config, 'hf_token', None)
        should_push = getattr(config, 'push_to_hub', False)
        try:
            self.api = HfApi(token=token) 
            print("DEBUG: HfApi initialized successfully.")
        except Exception as e:
            print(f"DEBUG: HfApi failed to initialize: {e}")
            self.api = None
        
    def setup_dirs(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.samples_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)

    def cleanup_local(self, prefix="full_state_", keep_last_n=5):
        search_pattern = os.path.join(self.checkpoint_dir, f"{prefix}*.pt")
        files = glob.glob(search_pattern)
        
        if len(files) <= keep_last_n:
            return

        def get_epoch_num(filepath):
            match = re.search(r"epoch_(\d+)", filepath)
            return int(match.group(1)) if match else -1

        files.sort(key=get_epoch_num)
        
        for f in files[:-keep_last_n]:
            try:
                os.remove(f)
                
                dir_name = os.path.dirname(f)
                file_name = os.path.basename(f)
                
                if "full_state_" in file_name:
                    ema_name = file_name.replace("full_state_", "ema_").replace("final_", "")
                else:
                    ema_name = file_name.replace("full_state", "ema")
                    
                ema_path = os.path.join(dir_name, ema_name)
                
                if os.path.exists(ema_path):
                    os.remove(ema_path)
                    
            except OSError as e:
                print(f"Error deleting checkpoint {f}: {e}")

    def save(self, epoch, global_step, model, ema_model, optimizer, scheduler, wandb_run_id, is_final=False):
        prefix = "full_state_final" if is_final else "full_state"
        
        state_filename = f"{prefix}_epoch_{epoch}.pt"
        ema_filename = f"ema_epoch_{epoch}.pt"
        
        save_path = os.path.join(self.checkpoint_dir, state_filename)
        ema_path = os.path.join(self.checkpoint_dir, ema_filename)
        
        full_state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema_model.module.state_dict() if hasattr(ema_model, 'module') else ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'wandb_run_id': wandb_run_id
        }
        torch.save(full_state, save_path)
        print(f"Saved Full State: {state_filename}")

        ema_state = ema_model.module.state_dict() if hasattr(ema_model, 'module') else ema_model.state_dict()
        torch.save(ema_state, ema_path)
        print(f"Saved EMA: {ema_filename}")
        
        if self.api:
            self._upload_to_hf(save_path, f"checkpoints/{state_filename}")
            self._upload_to_hf(ema_path, f"checkpoints/{ema_filename}")
            
            if not is_final:
                self._cleanup_hf(epoch)

        if not is_final:
            self.cleanup_local(prefix="full_state_", keep_last_n=5)

    def load_run_id(self, path):
        if not os.path.exists(path):
            return None
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False) 
            return checkpoint.get('wandb_run_id', None)
        except:
            return None

    def load(self, path, model, ema_model, optimizer=None, scheduler=None):
        if path == "latest":
            search_pattern = os.path.join(self.checkpoint_dir, "full_state_*.pt")
            files = glob.glob(search_pattern)
            if not files:
                print(f"No checkpoints found in {self.checkpoint_dir}")
                return 0, 0
            def get_epoch(f):
                match = re.search(r"epoch_(\d+)", f)
                return int(match.group(1)) if match else -1
            path = max(files, key=get_epoch)
            print(f"Resolved 'latest' to: {os.path.basename(path)}")

        if not os.path.exists(path):
            if self.api and self.config.hf_repo_id:
                print(f"Checkpoint not found locally. Attempting to download from HF: {self.config.hf_repo_id}")
                try:
                    filename = os.path.basename(path)
                    path = hf_hub_download(repo_id=self.config.hf_repo_id, filename=f"checkpoints/{filename}")
                    print(f"Successfully downloaded {filename}")
                except Exception as e:
                    print(f"HF Download failed: {e}")
                    return 0, 0
            else:
                print(f"Checkpoint not found: {path}")
                return 0, 0

        print(f"Resuming from {path}...")
        checkpoint = torch.load(path, map_location=self.config.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'ema_state_dict' in checkpoint:
            if hasattr(ema_model, 'module'):
                ema_model.module.load_state_dict(checkpoint['ema_state_dict'])
            else:
                ema_model.load_state_dict(checkpoint['ema_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                print(f"Optimizer load failed (likely architecture change). Resetting optimizer. Error: {e}")
        elif optimizer is None:
            print("Optimizer reset: Skipping state load.")
            
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif scheduler is None:
            print(">> Scheduler reset: Starting fresh.")
            
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        
        del checkpoint
        torch.cuda.empty_cache()
        
        return start_epoch, global_step
    
    def _upload_to_hf(self, local_path, path_in_repo):
        try:
            print(f"Uploading {os.path.basename(local_path)} to HF...")
            self.api.upload_file(
                path_or_fileobj=local_path, 
                path_in_repo=path_in_repo, 
                repo_id=self.config.hf_repo_id, 
                repo_type="model"
            )
        except Exception as e:
            print(f"Upload failed: {e}")

    def _cleanup_hf(self, current_epoch):
        keep_last = self.config.keep_last
        delete_epoch = current_epoch - (self.config.save_every * keep_last)
        
        if delete_epoch > 0:
            files_to_del = [
                f"checkpoints/full_state_epoch_{delete_epoch}.pt",
                f"checkpoints/ema_epoch_{delete_epoch}.pt"
            ]
            
            for file_path in files_to_del:
                try:
                    self.api.delete_file(file_path, repo_id=self.config.hf_repo_id)
                    print(f"Deleted old HF checkpoint: {file_path}")
                except Exception:
                    pass
                
    def _resolve_path(self, requested_path):
        if requested_path == "latest":
            search_pattern = os.path.join(self.checkpoint_dir, "full_state_*.pt")
            files = glob.glob(search_pattern)
            
            if not files and self.api and self.config.hf_repo_id:
                print("No local checkpoints. Checking HF for latest...")
                try:
                    repo_files = self.api.list_repo_files(repo_id=self.config.hf_repo_id)
                    checkpoints =[f for f in repo_files if f.startswith("checkpoints/full_state_epoch_")]
                    
                    if checkpoints:
                        def get_epoch(f):
                            match = re.search(r"epoch_(\d+)", f)
                            return int(match.group(1)) if match else -1
                        latest_hf_file = max(checkpoints, key=get_epoch)
                        return latest_hf_file.replace("checkpoints/", "") 
                except Exception as e:
                    print(f"Failed to fetch latest from HF: {e}")
            
            if not files:
                print(f"No full_state checkpoints found locally or on HF.")
                return None
                
            def get_epoch_num(filepath):
                match = re.search(r"epoch_(\d+)", filepath)
                return int(match.group(1)) if match else -1
            latest_file = max(files, key=get_epoch_num)
            return latest_file
            
        elif os.path.exists(requested_path):
            return requested_path
        else:
            return requested_path