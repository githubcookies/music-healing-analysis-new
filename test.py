import os
import numpy as np
from healing_features import HealingFeatures

def test_music_files():
    analyzer = HealingFeatures()
    
    # 测试文件路径
    healing_dir = "healing_music"
    non_healing_dir = "non_healing_music"
    
    # 获取测试文件
    healing_files = [f for f in os.listdir(healing_dir) if f.endswith(('.mp3', '.wav', '.flac', '.m4a'))][:3]
    non_healing_files = [f for f in os.listdir(non_healing_dir) if f.endswith(('.mp3', '.wav', '.flac', '.m4a'))][:3]
    
    # 存储所有得分
    healing_scores = []
    non_healing_scores = []
    
    print("\n=== 治愈音乐分析结果 ===")
    print("-" * 50)
    for file in healing_files:
        file_path = os.path.join(healing_dir, file)
        scores = analyzer.calculate_healing_score(file_path)
        healing_scores.append(scores)
        
        print(f"\n文件名: {file}")
        print(f"总分: {scores['total_score']:.2f}")
        print(f"粗糙度得分: {scores['roughness_score']:.2f}")
        print(f"音程亲和力得分: {scores['consonance_score']:.2f}")
        print(f"频谱能量得分: {scores['spectral_score']:.2f}")
        print("-" * 30)
    
    print("\n=== 非治愈音乐分析结果 ===")
    print("-" * 50)
    for file in non_healing_files:
        file_path = os.path.join(non_healing_dir, file)
        scores = analyzer.calculate_healing_score(file_path)
        non_healing_scores.append(scores)
        
        print(f"\n文件名: {file}")
        print(f"总分: {scores['total_score']:.2f}")
        print(f"粗糙度得分: {scores['roughness_score']:.2f}")
        print(f"音程亲和力得分: {scores['consonance_score']:.2f}")
        print(f"频谱能量得分: {scores['spectral_score']:.2f}")
        print("-" * 30)
    
    # 计算平均分数
    print("\n=== 平均分数对比 ===")
    print("-" * 50)
    
    def calculate_averages(scores_list):
        total = np.mean([s['total_score'] for s in scores_list])
        roughness = np.mean([s['roughness_score'] for s in scores_list])
        consonance = np.mean([s['consonance_score'] for s in scores_list])
        spectral = np.mean([s['spectral_score'] for s in scores_list])
        return total, roughness, consonance, spectral
    
    h_total, h_rough, h_cons, h_spec = calculate_averages(healing_scores)
    nh_total, nh_rough, nh_cons, nh_spec = calculate_averages(non_healing_scores)
    
    print("\n治愈音乐平均分：")
    print(f"总分: {h_total:.2f}")
    print(f"粗糙度得分: {h_rough:.2f}")
    print(f"音程亲和力得分: {h_cons:.2f}")
    print(f"频谱能量得分: {h_spec:.2f}")
    
    print("\n非治愈音乐平均分：")
    print(f"总分: {nh_total:.2f}")
    print(f"粗糙度得分: {nh_rough:.2f}")
    print(f"音程亲和力得分: {nh_cons:.2f}")
    print(f"频谱能量得分: {nh_spec:.2f}")
    
    print("\n分数差异（治愈 - 非治愈）：")
    print(f"总分差异: {(h_total - nh_total):.2f}")
    print(f"粗糙度得分差异: {(h_rough - nh_rough):.2f}")
    print(f"音程亲和力得分差异: {(h_cons - nh_cons):.2f}")
    print(f"频谱能量得分差异: {(h_spec - nh_spec):.2f}")

if __name__ == "__main__":
    test_music_files() 