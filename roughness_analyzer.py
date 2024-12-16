import librosa
import numpy as np

class RoughnessAnalyzer:
    def __init__(self):
        pass
    
    def calculate_roughness(self, audio_path):
        """计算音频的谐波粗糙度（优化版本）"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, duration=30)
            
            # 计算频谱质心
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            
            # 计算频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            
            # 计算谐波和打击乐成分
            harmonic, percussive = librosa.effects.hpss(y)
            harmonic_energy = np.mean(np.abs(harmonic))
            percussive_energy = np.mean(np.abs(percussive))
            
            # 计算色谱图特征
            chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)
            chroma_var = np.var(chroma)
            
            # 调整权重计算
            centroid_score = 1 - (np.mean(spectral_centroid) / (sr/4))  # 降低高频成分的影响
            bandwidth_score = 1 - (np.mean(spectral_bandwidth) / (sr/4))  # 降低频谱带宽的影响
            harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-6)
            chroma_score = 1 - np.clip(chroma_var, 0, 1)  # 音高变化的平稳性
            
            # 新的权重分配
            roughness_score = (
                harmonic_ratio * 0.4 +          # 谐波占比权重增加
                centroid_score * 0.2 +          # 频谱质心
                bandwidth_score * 0.2 +         # 频谱带宽
                chroma_score * 0.2              # 音高稳定性
            )
            
            # 使用sigmoid函数使分布更加极化
            def sigmoid(x, k=8):
                return 1 / (1 + np.exp(-k * (x - 0.5)))
            
            # 应用sigmoid函数并确保得分在0-1范围内
            final_score = np.clip(sigmoid(roughness_score), 0, 1)
            
            return float(final_score)
            
        except Exception as e:
            print(f"粗糙度计算出错: {str(e)}")
            return 0.5  # 返回默认值