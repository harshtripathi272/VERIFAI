
import sys
import os

# Add parent dir to path
sys.path.append(os.getcwd())

import qlora_medgemma
from qlora_medgemma import main, TrainingConfig

# Monkeypatch Config to be fast
def mock_init(self):
    self.model = qlora_medgemma.ModelConfig()
    self.lora = qlora_medgemma.LoraConfigData()
    self.data = qlora_medgemma.DataConfig()
    self.data.max_images = 2 # Speed up
    
    self.output_dir = "test_output"
    self.num_train_epochs = 0.01
    self.per_device_train_batch_size = 1
    self.per_device_eval_batch_size = 1
    self.gradient_accumulation_steps = 1
    self.learning_rate = 2e-4
    self.weight_decay = 0.01
    self.max_grad_norm = 0.3
    self.warmup_ratio = 0.03
    self.lr_scheduler_type = "linear"
    self.gradient_checkpointing = False # Faster startup
    self.optim = "adamw_torch"
    self.logging_steps = 1
    self.save_steps = 10
    self.eval_steps = 10
    self.seed = 42
    self.bf16 = False
    self.fp16 = True
    self.report_to = "none"
    self.run_version = "dryrun"
    self.max_train_samples = 5 # VERY SMALL

qlora_medgemma.TrainingConfig.__init__ = mock_init

if __name__ == "__main__":
    print("Running dry-run...")
    # main() 
    # Calling main() will run training. 
    # We just want to see if it starts.
    try:
        main()
        print("Dry run finished successfully!")
    except Exception as e:
        print(f"Dry run failed: {e}")
        import traceback
        traceback.print_exc()
