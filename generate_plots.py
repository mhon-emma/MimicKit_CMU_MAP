#!/usr/bin/env python3
"""
Generate PNG plots from training log for write-up.
Usage: python generate_plots.py output/11_18_with_constant_force_g.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5

def parse_training_log(filepath):
    """Parse the training log file into a pandas DataFrame."""
    print(f"Reading file: {filepath}")
    df = pd.read_csv(filepath, sep=r'\s+', engine='python')
    return df

def create_figure1_episode_length(df, save_dir):
    """Figure 1: Episode Length over training."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df['Iteration'], df['Test_Episode_Length'],
            label='Test', linewidth=2.5, color='#2E86AB', alpha=0.9)
    ax.plot(df['Iteration'], df['Train_Episode_Length'],
            label='Train', linewidth=2.5, color='#A23B72', alpha=0.9)

    # Highlight collapse region
    ax.axvspan(3000, 7000, alpha=0.15, color='red', label='Mid-Training Collapse')

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Length (steps)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 1: Episode Length Progression', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_dir / 'figure1_episode_length.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure1_episode_length.png")
    plt.close()

def create_figure2_discriminator_metrics(df, save_dir):
    """Figure 2: Discriminator Metrics (2 panels)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Panel A: Accuracies
    ax1.plot(df['Iteration'], df['Disc_Agent_Acc'],
            label='Agent Accuracy', linewidth=2.5, color='#2E86AB')
    ax1.plot(df['Iteration'], df['Disc_Demo_Acc'],
            label='Demo Accuracy', linewidth=2.5, color='#F24236')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Discriminator Accuracy', fontsize=11, fontweight='bold', loc='left')
    ax1.legend(loc='best', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.set_ylim([0, 1.05])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Reward Mean with Std shading
    ax2.plot(df['Iteration'], df['Disc_Reward_Mean'],
            linewidth=2.5, color='#06A77D', label='Reward Mean')
    ax2.fill_between(df['Iteration'],
                     df['Disc_Reward_Mean'] - df['Disc_Reward_Std'],
                     df['Disc_Reward_Mean'] + df['Disc_Reward_Std'],
                     alpha=0.25, color='#06A77D', label='± 1 Std')
    ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Discriminator Reward', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Discriminator Reward', fontsize=11, fontweight='bold', loc='left')
    ax2.legend(loc='best', fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Figure 2: Discriminator Metrics', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_dir / 'figure2_discriminator_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure2_discriminator_metrics.png")
    plt.close()

def create_figure3_loss_curves(df, save_dir):
    """Figure 3: Loss Curves (2x2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: Total Loss (log scale)
    axes[0, 0].semilogy(df['Iteration'], df['Loss'],
                        linewidth=2.5, color='#2E86AB')
    axes[0, 0].set_ylabel('Total Loss (log)', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('(A) Total Loss', fontsize=10, fontweight='bold', loc='left')
    axes[0, 0].grid(True, alpha=0.25, linestyle='--')
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)

    # Panel B: Critic Loss
    axes[0, 1].plot(df['Iteration'], df['Critic_Loss'],
                    linewidth=2.5, color='#A23B72')
    axes[0, 1].set_ylabel('Critic Loss', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('(B) Critic Loss', fontsize=10, fontweight='bold', loc='left')
    axes[0, 1].grid(True, alpha=0.25, linestyle='--')
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)

    # Panel C: Actor Loss
    axes[1, 0].plot(df['Iteration'], df['Actor_Loss'],
                    linewidth=2.5, color='#F24236')
    axes[1, 0].set_xlabel('Iteration', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Actor Loss', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('(C) Actor Loss', fontsize=10, fontweight='bold', loc='left')
    axes[1, 0].grid(True, alpha=0.25, linestyle='--')
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    # Panel D: Discriminator Loss
    axes[1, 1].plot(df['Iteration'], df['Disc_Loss'],
                    linewidth=2.5, color='#06A77D')
    axes[1, 1].set_xlabel('Iteration', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Discriminator Loss', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('(D) Discriminator Loss', fontsize=10, fontweight='bold', loc='left')
    axes[1, 1].grid(True, alpha=0.25, linestyle='--')
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)

    fig.suptitle('Figure 3: Loss Curves', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_dir / 'figure3_loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure3_loss_curves.png")
    plt.close()

def create_figure4_discriminator_logits(df, save_dir):
    """Figure 4: Discriminator Logits."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df['Iteration'], df['Disc_Agent_Logit'],
            label='Agent Logits', linewidth=2.5, color='#2E86AB', alpha=0.9)
    ax.plot(df['Iteration'], df['Disc_Demo_Logit'],
            label='Demo Logits', linewidth=2.5, color='#F24236', alpha=0.9)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Logit Value', fontsize=12, fontweight='bold')
    ax.set_title('Figure 4: Discriminator Logits (Agent vs Demo)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_dir / 'figure4_discriminator_logits.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure4_discriminator_logits.png")
    plt.close()

def create_figure5_training_efficiency(df, save_dir):
    """Figure 5: Training Efficiency."""
    fig, ax = plt.subplots(figsize=(10, 5))

    wall_time_hours = df['Wall_Time'] / 3600
    ax.plot(wall_time_hours, df['Train_Episode_Length'],
            linewidth=2.5, color='#2E86AB', alpha=0.9)

    ax.set_xlabel('Wall Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Train Episode Length', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5: Training Efficiency (Episode Length vs Time)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_dir / 'figure5_training_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure5_training_efficiency.png")
    plt.close()

def create_figure6_ppo_diagnostics(df, save_dir):
    """Figure 6: PPO Diagnostics (2x2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: Advantage Mean with Std
    axes[0, 0].plot(df['Iteration'], df['Adv_Mean'],
                    linewidth=2.5, color='#2E86AB', label='Mean')
    axes[0, 0].fill_between(df['Iteration'],
                           df['Adv_Mean'] - df['Adv_Std'],
                           df['Adv_Mean'] + df['Adv_Std'],
                           alpha=0.25, color='#2E86AB', label='± 1 Std')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
    axes[0, 0].set_ylabel('Advantage', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('(A) Advantage', fontsize=10, fontweight='bold', loc='left')
    axes[0, 0].legend(loc='best', fontsize=8, framealpha=0.95)
    axes[0, 0].grid(True, alpha=0.25, linestyle='--')
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)

    # Panel B: Clip Fraction
    axes[0, 1].plot(df['Iteration'], df['Clip_Frac'],
                    linewidth=2.5, color='#A23B72')
    axes[0, 1].set_ylabel('Clip Fraction', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('(B) Clip Fraction', fontsize=10, fontweight='bold', loc='left')
    axes[0, 1].grid(True, alpha=0.25, linestyle='--')
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)

    # Panel C: Importance Ratio
    axes[1, 0].plot(df['Iteration'], df['Imp_Ratio'],
                    linewidth=2.5, color='#F24236')
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
    axes[1, 0].set_xlabel('Iteration', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Importance Ratio', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('(C) Importance Ratio', fontsize=10, fontweight='bold', loc='left')
    axes[1, 0].grid(True, alpha=0.25, linestyle='--')
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    # Panel D: Action Bound Loss
    axes[1, 1].plot(df['Iteration'], df['Action_Bound_Loss'],
                    linewidth=2.5, color='#06A77D')
    axes[1, 1].set_xlabel('Iteration', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Action Bound Loss', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('(D) Action Bound Loss', fontsize=10, fontweight='bold', loc='left')
    axes[1, 1].grid(True, alpha=0.25, linestyle='--')
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)

    fig.suptitle('Figure 6: PPO Diagnostics', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_dir / 'figure6_ppo_diagnostics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure6_ppo_diagnostics.png")
    plt.close()

def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Iterations:       {df['Iteration'].max():,}")
    print(f"Samples:          {df['Samples'].max():,}")
    print(f"Wall Time:        {df['Wall_Time'].max()/3600:.1f} hours")
    print(f"Episode Length:   {df['Test_Episode_Length'].iloc[0]:.1f} → {df['Test_Episode_Length'].iloc[-1]:.1f} (test)")
    print(f"                  {df['Train_Episode_Length'].iloc[0]:.1f} → {df['Train_Episode_Length'].iloc[-1]:.1f} (train)")
    print(f"Total Loss:       {df['Loss'].iloc[0]:.2f} → {df['Loss'].iloc[-1]:.2f} ({(1-df['Loss'].iloc[-1]/df['Loss'].iloc[0])*100:.1f}% reduction)")
    print("="*70 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <path_to_log_file>")
        print("Example: python generate_plots.py output/11_18_with_constant_force_g.txt")
        sys.exit(1)

    log_file = Path(sys.argv[1])

    if not log_file.exists():
        print(f"Error: File not found: {log_file}")
        sys.exit(1)

    print(f"\n📊 Generating plots from: {log_file.name}")

    try:
        df = parse_training_log(log_file)
        print(f"✓ Loaded {len(df)} iterations")
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create output directory
    save_dir = log_file.parent / f"{log_file.stem}_figures"
    save_dir.mkdir(exist_ok=True)
    print(f"\n📁 Saving figures to: {save_dir}")

    # Generate all figures
    print("\n🎨 Generating publication-quality figures...")
    create_figure1_episode_length(df, save_dir)
    create_figure2_discriminator_metrics(df, save_dir)
    create_figure3_loss_curves(df, save_dir)
    create_figure4_discriminator_logits(df, save_dir)
    create_figure5_training_efficiency(df, save_dir)
    create_figure6_ppo_diagnostics(df, save_dir)

    # Print summary
    print_summary_statistics(df)

    print(f"✅ All figures saved to: {save_dir}\n")

if __name__ == "__main__":
    main()
