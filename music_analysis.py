import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from healing_features import HealingFeatures

def analyze_music_files(base_path='.'):
    """分析音乐文件并生成评分"""
    healing_dir = os.path.join(base_path, 'healing_music')
    non_healing_dir = os.path.join(base_path, 'non_healing_music')
    
    # 获取所有音频文件
    healing_files = [f for f in os.listdir(healing_dir) if f.endswith(('.mp3', '.wav', '.flac', '.m4a'))]
    non_healing_files = [f for f in os.listdir(non_healing_dir) if f.endswith(('.mp3', '.wav', '.flac', '.m4a'))]
    
    analyzer = HealingFeatures()
    results = []
    
    def process_file(file_path, is_healing):
        try:
            scores = analyzer.calculate_healing_score(file_path)
            return {
                'filename': os.path.basename(file_path),
                'is_healing': is_healing,
                'total_score': float(scores['total_score']),
                'roughness_score': float(scores['roughness_score']),
                'consonance_score': float(scores['consonance_score']),
                'spectral_score': float(scores['spectral_score'])
            }
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None
    
    print("\nAnalyzing healing music...")
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, os.path.join(healing_dir, f), True)
            for f in healing_files
        ]
        for future in tqdm(futures, total=len(healing_files)):
            result = future.result()
            if result:
                results.append(result)
    
    print("\nAnalyzing non-healing music...")
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, os.path.join(non_healing_dir, f), False)
            for f in non_healing_files
        ]
        for future in tqdm(futures, total=len(non_healing_files)):
            result = future.result()
            if result:
                results.append(result)
    
    # Convert to DataFrame and ensure numeric types
    df = pd.DataFrame(results)
    numeric_columns = ['total_score', 'roughness_score', 'consonance_score', 'spectral_score']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate average scores
    healing_avg = df[df['is_healing']][numeric_columns].mean()
    non_healing_avg = df[~df['is_healing']][numeric_columns].mean()
    
    print("\n=== Average Score Comparison ===")
    print("-" * 50)
    print("\nHealing Music Averages:")
    print(f"Total Score: {healing_avg['total_score']:.2f}")
    print(f"Roughness Score: {healing_avg['roughness_score']:.2f}")
    print(f"Consonance Score: {healing_avg['consonance_score']:.2f}")
    print(f"Spectral Score: {healing_avg['spectral_score']:.2f}")
    
    print("\nNon-healing Music Averages:")
    print(f"Total Score: {non_healing_avg['total_score']:.2f}")
    print(f"Roughness Score: {non_healing_avg['roughness_score']:.2f}")
    print(f"Consonance Score: {non_healing_avg['consonance_score']:.2f}")
    print(f"Spectral Score: {non_healing_avg['spectral_score']:.2f}")
    
    print("\nScore Differences (Healing - Non-healing):")
    score_diff = healing_avg - non_healing_avg
    print(f"Total Score Difference: {score_diff['total_score']:.2f}")
    print(f"Roughness Score Difference: {score_diff['roughness_score']:.2f}")
    print(f"Consonance Score Difference: {score_diff['consonance_score']:.2f}")
    print(f"Spectral Score Difference: {score_diff['spectral_score']:.2f}")
    
    # Save results
    df.to_excel('analysis_results.xlsx', index=False)
    print("\nAnalysis results saved to analysis_results.xlsx")
    
    # Set matplotlib style
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 10))
    
    # Plot distributions
    plt.subplot(2, 2, 1)
    plt.hist(df[df['is_healing']]['total_score'].dropna(), alpha=0.5, label='Healing Music', bins=20, color='lightgreen')
    plt.hist(df[~df['is_healing']]['total_score'].dropna(), alpha=0.5, label='Non-healing Music', bins=20, color='lightcoral')
    plt.title('Total Score Distribution', fontsize=12)
    plt.xlabel('Score', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.legend(fontsize=10)
    
    plt.subplot(2, 2, 2)
    plt.hist(df[df['is_healing']]['roughness_score'].dropna(), alpha=0.5, label='Healing Music', bins=20, color='lightgreen')
    plt.hist(df[~df['is_healing']]['roughness_score'].dropna(), alpha=0.5, label='Non-healing Music', bins=20, color='lightcoral')
    plt.title('Roughness Score Distribution', fontsize=12)
    plt.xlabel('Score', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.legend(fontsize=10)
    
    plt.subplot(2, 2, 3)
    plt.hist(df[df['is_healing']]['consonance_score'].dropna(), alpha=0.5, label='Healing Music', bins=20, color='lightgreen')
    plt.hist(df[~df['is_healing']]['consonance_score'].dropna(), alpha=0.5, label='Non-healing Music', bins=20, color='lightcoral')
    plt.title('Consonance Score Distribution', fontsize=12)
    plt.xlabel('Score', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.legend(fontsize=10)
    
    plt.subplot(2, 2, 4)
    plt.hist(df[df['is_healing']]['spectral_score'].dropna(), alpha=0.5, label='Healing Music', bins=20, color='lightgreen')
    plt.hist(df[~df['is_healing']]['spectral_score'].dropna(), alpha=0.5, label='Non-healing Music', bins=20, color='lightcoral')
    plt.title('Spectral Score Distribution', fontsize=12)
    plt.xlabel('Score', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.legend(fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout(pad=2.0)
    plt.savefig('score_distributions.png', dpi=300, bbox_inches='tight')
    print("Score distributions saved to score_distributions.png")

if __name__ == "__main__":
    analyze_music_files() 