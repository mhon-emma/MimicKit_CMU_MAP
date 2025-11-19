#!/usr/bin/env python3
"""
Convert training log text file to TensorBoard format.
Usage: python convert_to_tensorboard.py output/11_18_with_constant_force_g.txt
"""

import pandas as pd
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def parse_training_log(filepath):
    """Parse the training log file into a pandas DataFrame."""
    print(f"Reading file: {filepath}")
    df = pd.read_csv(filepath, sep=r'\s+', engine='python')
    print(f"Columns found: {list(df.columns)[:5]}...")
    return df

def convert_to_tensorboard(df, log_dir):
    """Convert DataFrame to TensorBoard format."""
    writer = SummaryWriter(log_dir=log_dir)

    print(f"\n📝 Writing {len(df)} iterations to TensorBoard...")

    for idx, row in df.iterrows():
        iteration = int(row['Iteration'])

        # ==== FIGURE 1: Episode Length (Main Result) ====
        writer.add_scalars('1_Episode_Length/Combined', {
            'Test': row['Test_Episode_Length'],
            'Train': row['Train_Episode_Length']
        }, iteration)

        # ==== FIGURE 2: Discriminator Metrics ====
        # Panel A: Accuracies
        writer.add_scalars('2_Discriminator/A_Accuracies', {
            'Agent_Accuracy': row['Disc_Agent_Acc'],
            'Demo_Accuracy': row['Disc_Demo_Acc']
        }, iteration)

        # Panel B: Rewards
        writer.add_scalar('2_Discriminator/B_Reward_Mean', row['Disc_Reward_Mean'], iteration)
        writer.add_scalar('2_Discriminator/B_Reward_Std', row['Disc_Reward_Std'], iteration)

        # ==== FIGURE 3: Loss Curves ====
        writer.add_scalar('3_Loss/A_Total_Loss', row['Loss'], iteration)
        writer.add_scalar('3_Loss/B_Critic_Loss', row['Critic_Loss'], iteration)
        writer.add_scalar('3_Loss/C_Actor_Loss', row['Actor_Loss'], iteration)
        writer.add_scalar('3_Loss/D_Discriminator_Loss', row['Disc_Loss'], iteration)

        # ==== FIGURE 4: Discriminator Logits ====
        writer.add_scalars('4_Discriminator_Logits/Combined', {
            'Agent_Logit': row['Disc_Agent_Logit'],
            'Demo_Logit': row['Disc_Demo_Logit']
        }, iteration)

        # ==== FIGURE 5: Training Efficiency ====
        writer.add_scalar('5_Training_Efficiency/Episode_Length_vs_Time',
                         row['Train_Episode_Length'], iteration)
        writer.add_scalar('5_Training_Efficiency/Wall_Time_Hours',
                         row['Wall_Time'] / 3600, iteration)

        # ==== FIGURE 6: PPO Diagnostics ====
        # Panel A: Advantages
        writer.add_scalar('6_PPO_Diagnostics/A_Advantage_Mean', row['Adv_Mean'], iteration)
        writer.add_scalar('6_PPO_Diagnostics/A_Advantage_Std', row['Adv_Std'], iteration)

        # Panel B: Clip Fraction
        writer.add_scalar('6_PPO_Diagnostics/B_Clip_Fraction', row['Clip_Frac'], iteration)

        # Panel C: Importance Ratio
        writer.add_scalar('6_PPO_Diagnostics/C_Importance_Ratio', row['Imp_Ratio'], iteration)

        # Panel D: Action Bound Loss
        writer.add_scalar('6_PPO_Diagnostics/D_Action_Bound_Loss', row['Action_Bound_Loss'], iteration)

        # ==== Additional Metrics (not for main figures) ====
        writer.add_scalar('Additional/Test_Return', row['Test_Return'], iteration)
        writer.add_scalar('Additional/Train_Return', row['Train_Return'], iteration)
        writer.add_scalar('Additional/Samples', row['Samples'], iteration)
        writer.add_scalar('Additional/Test_Episodes', row['Test_Episodes'], iteration)
        writer.add_scalar('Additional/Train_Episodes', row['Train_Episodes'], iteration)

        # Discriminator details
        writer.add_scalar('Additional/Disc_Grad_Penalty', row['Disc_Grad_Penalty'], iteration)
        writer.add_scalar('Additional/Disc_Logit_Loss', row['Disc_Logit_Loss'], iteration)
        writer.add_scalar('Additional/Disc_Weight_Decay', row['Disc_Weight_Decay'], iteration)

        # Observation normalization
        writer.add_scalar('Additional/Obs_Norm_Mean', row['Obs_Norm_Mean'], iteration)
        writer.add_scalar('Additional/Obs_Norm_Std', row['Obs_Norm_Std'], iteration)
        writer.add_scalar('Additional/Exp_Prob', row['Exp_Prob'], iteration)

        if (idx + 1) % 50 == 0:
            print(f"  ✓ Written {idx + 1}/{len(df)} iterations")

    writer.close()
    print(f"\n✅ TensorBoard logs written to: {log_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_tensorboard.py <path_to_log_file>")
        print("Example: python convert_to_tensorboard.py output/11_18_with_constant_force_g.txt")
        sys.exit(1)

    log_file = Path(sys.argv[1])

    if not log_file.exists():
        print(f"Error: File not found: {log_file}")
        sys.exit(1)

    print(f"\n📊 Converting training log: {log_file.name}")

    try:
        df = parse_training_log(log_file)
        print(f"✓ Loaded {len(df)} iterations")
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create TensorBoard log directory
    tb_log_dir = log_file.parent / f"tensorboard_{log_file.stem}"
    tb_log_dir.mkdir(exist_ok=True)

    # Convert to TensorBoard format
    convert_to_tensorboard(df, str(tb_log_dir))

    print("\n🚀 To view in TensorBoard, run:")
    print(f"   tensorboard --logdir {tb_log_dir} --bind_all")
    print()

if __name__ == "__main__":
    main()
