import os
import glob
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(results_dir, out_dir):
    """
    Load metrics from results_dir/models and create visualizations in out_dir
    """
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(out_dir, exist_ok=True)

    # Plot training curves (loss, accuracy, F1) for each epoch
    epoch_files = sorted(glob.glob(os.path.join(models_dir, '*_epoch*_metrics.json')))
    if epoch_files:
        records = []
        for f in epoch_files:
            fname = os.path.basename(f)
            parts = fname.split('_epoch')
            run = parts[0]
            epoch = int(parts[1].split('_')[0])
            with open(f) as fp:
                metrics = json.load(fp)
            records.append({
                'run': run,
                'epoch': epoch,
                'val_loss': metrics.get('val_loss'),
                'val_accuracy': metrics.get('val_accuracy'),
                'val_f1': metrics.get('val_f1')
            })
        df = pd.DataFrame(records)
        sns.set(style='whitegrid')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.lineplot(data=df, x='epoch', y='val_loss', hue='run', marker='o', ax=axes[0])
        axes[0].set_title('Validation Loss')
        sns.lineplot(data=df, x='epoch', y='val_accuracy', hue='run', marker='o', ax=axes[1])
        axes[1].set_title('Validation Accuracy')
        sns.lineplot(data=df, x='epoch', y='val_f1', hue='run', marker='o', ax=axes[2])
        axes[2].set_title('Validation F1 Score')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'training_curves.png'))
        plt.close()

    # Plot final test metrics across runs
    final_files = sorted(glob.glob(os.path.join(models_dir, '*_metrics.json')))
    recs = []
    for f in final_files:
        fname = os.path.basename(f)
        if '_epoch' in fname:
            continue
        run = fname.replace('_metrics.json', '')
        with open(f) as fp:
            metrics = json.load(fp)
        recs.append({'run': run, **metrics})
    if recs:
        df_final = pd.DataFrame(recs)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df_final, x='run', y='test_f1')
        plt.xticks(rotation=45, ha='right')
        plt.title('Test F1 per Run')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'test_f1_per_run.png'))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results', help='Path to results directory')
    parser.add_argument('--out_dir', type=str, default=os.path.join('results', 'visualizations'), help='Output directory for visualizations')
    args = parser.parse_args()
    main(args.results_dir, args.out_dir)
