"""
TensorBoard Export & Observability Suite for CO3133 Assignment 01
Exports epoch-by-epoch training metrics for all 5 architectures into ./runs/
Tracks:
- Loss (Train vs Val)
- Generalization Gap (Val Loss - Train Loss) -> High Variance Diagnosis
- Bias Indicator (Training Loss floor) -> High Bias Diagnosis
- Accuracy & Macro-F1 (Train vs Val)
- Learning Rate Schedule
"""

import os
import json
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def export_histories_to_tensorboard(log_dir="./runs", history_dir="./outputs/checkpoints"):
    os.makedirs(log_dir, exist_ok=True)
    
    models = ["linear", "mlp", "cnn", "lstm", "vit"]
    print(f"Exporting training histories from {history_dir} to TensorBoard at {log_dir}...")
    
    for m_name in models:
        json_path = os.path.join(history_dir, f"{m_name}_history.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Skipping {m_name}.")
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            h = json.load(f)
            
        model_log_dir = os.path.join(log_dir, m_name.upper())
        writer = SummaryWriter(log_dir=model_log_dir)
        
        num_epochs = len(h.get("train_loss", []))
        for epoch in range(num_epochs):
            step = epoch + 1
            
            # Loss metrics
            tr_loss = h["train_loss"][epoch]
            val_loss = h["val_loss"][epoch]
            gen_gap = val_loss - tr_loss  # Key metric to diagnose Variance / Overfitting!
            
            writer.add_scalar("Loss/Train", tr_loss, step)
            writer.add_scalar("Loss/Val", val_loss, step)
            writer.add_scalar("Diagnosis/Generalization_Gap (Variance)", gen_gap, step)
            writer.add_scalar("Diagnosis/Train_Loss_Floor (Bias)", tr_loss, step)
            
            # Accuracy & Macro-F1
            tr_acc = h["train_acc"][epoch]
            val_acc = h["val_acc"][epoch]
            tr_f1 = h["train_macro_f1"][epoch]
            val_f1 = h["val_macro_f1"][epoch]
            
            writer.add_scalar("Accuracy/Train", tr_acc, step)
            writer.add_scalar("Accuracy/Val", val_acc, step)
            writer.add_scalar("Macro_F1/Train", tr_f1, step)
            writer.add_scalar("Macro_F1/Val", val_f1, step)
            
            # Learning rate
            if "lr" in h and epoch < len(h["lr"]):
                writer.add_scalar("Optimization/Learning_Rate", h["lr"][epoch], step)
                
        writer.flush()
        writer.close()
        print(f"  -> Exported {m_name.upper():<7} ({num_epochs} epochs) to {model_log_dir}")
        
    print(f"\nTensorBoard event logs ready!")
    print(f"To launch the interactive dashboard, run:")
    print(f"    tensorboard --logdir={log_dir} --port=6006")

if __name__ == "__main__":
    export_histories_to_tensorboard()
