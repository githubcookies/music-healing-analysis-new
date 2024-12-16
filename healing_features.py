import numpy as np
import librosa
from scipy.stats import entropy
from scipy.signal import spectrogram
import warnings
warnings.filterwarnings('ignore')

def sigmoid_transform(x, k=0.15, x0=65):
    """
    Applies a modified sigmoid transformation to stretch the score distribution
    k: controls the steepness of the curve
    x0: the midpoint of the sigmoid (inflection point)
    """
    return 100 / (1 + np.exp(-k * (x - x0)))

class HealingFeatures:
    def __init__(self):
        self.sample_rate = 22050
        self.frame_length = 2048
        self.hop_length = 512
        
        # 定义音程权重系统
        # 完全协和音程
        self.perfect_consonant_weights = {
            0: 1.0,   # 纯一度
            12: 1.0,  # 纯八度
            7: 0.9,   # 纯五度
            5: 0.8,   # 纯四度
        }
        
        # 不完全协和音程
        self.imperfect_consonant_weights = {
            4: 0.6,   # 大三度
            3: 0.5,   # 小三度
            9: 0.4,   # 大六度
            8: 0.3,   # 小六度
        }
        
        # 不协和音程
        self.dissonant_weights = {
            2: -0.2,  # 大二度
            1: -0.4,  # 小二度
            6: -0.6,  # 增四度/减五度
            10: -0.3, # 小七度
            11: -0.5, # 大七度
        }
        
        # 合并所有权重
        self.interval_weights = {
            **self.perfect_consonant_weights,
            **self.imperfect_consonant_weights,
            **self.dissonant_weights
        }

    def extract_pitch(self, y):
        """提取音频的音高序列"""
        try:
            # 设置更保守的参数
            hop_length = 512
            fmin = librosa.note_to_hz('C2')
            fmax = librosa.note_to_hz('C7')
            
            # 使用更稳定的音高检测方法
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=self.sample_rate, 
                hop_length=hop_length,
                fmin=fmin, 
                fmax=fmax,
                threshold=0.1
            )
            
            # 使用更安全的方式选择音高
            pitch_sequence = np.zeros_like(magnitudes[0])
            for i in range(magnitudes.shape[1]):
                max_index = magnitudes[:, i].argmax()
                if magnitudes[max_index, i] > 0:
                    pitch_sequence[i] = pitches[max_index, i]
            
            return pitch_sequence
            
        except Exception as e:
            print(f"音高提取出错: {str(e)}")
            return np.zeros(1024)

    def calculate_intervals(self, pitch_sequence):
        """计算音程序列"""
        try:
            # 过滤有效的音高值
            valid_pitches = pitch_sequence[pitch_sequence > 0]
            
            if len(valid_pitches) < 2:
                return np.array([])
            
            # 转换为MIDI音符号
            midi_notes = librosa.hz_to_midi(valid_pitches)
            
            # 计算音程并滤异常值
            intervals = np.abs(np.diff(midi_notes))
            intervals = intervals[intervals <= 24]  # 限制在两个八度内
            intervals = intervals[~np.isnan(intervals)]  # 移除NaN值
            
            return intervals
            
        except Exception as e:
            print(f"音程计算出错: {str(e)}")
            return np.array([])

    def calculate_consonance(self, y):
        """计算音程亲和力分数"""
        try:
            # 提取音高序列
            pitch_sequence = self.extract_pitch(y)
            
            # 计算音程
            intervals = self.calculate_intervals(pitch_sequence)
            
            if len(intervals) == 0:
                return 60.0
            
            # 计算加权得分
            total_weight = 0
            for interval in intervals:
                interval_int = int(round(interval)) % 12  # 将音程映射到一个八度内
                weight = self.interval_weights.get(interval_int, 0)
                total_weight += weight
            
            # 计算平均权重
            avg_weight = total_weight / len(intervals)
            
            # 归一化到0-100范围
            min_score = -0.6  # 最不和谐的权重
            max_score = 1.0   # 最和谐的权重
            normalized_score = (avg_weight - min_score) / (max_score - min_score) * 100
            
            # 限制在0-100范围内
            return float(np.clip(normalized_score, 0, 100))
            
        except Exception as e:
            print(f"和声计算出错: {str(e)}")
            return 60.0

    def calculate_roughness(self, y):
        # 计算频谱
        D = librosa.stft(y, n_fft=self.frame_length, hop_length=self.hop_length)
        S = np.abs(D)
        
        # 计算频谱的时间导数
        spectral_diff = np.diff(S, axis=1)
        
        # 计算粗糙度得分
        roughness = np.mean(np.abs(spectral_diff))
        
        # 归一化到0-100
        max_roughness = 2.0  # 根据经验设定的最大粗糙度值
        roughness_score = 100 * (1 - min(roughness / max_roughness, 1))
        
        return roughness_score

    def calculate_spectral_features(self, y):
        try:
            # 计算梅尔频谱图
            mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 计算频谱质心并确保有限值
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)[0]
            spectral_centroids = np.nan_to_num(spectral_centroids, nan=0.0, posinf=self.sample_rate/2)
            
            # 计算频谱带宽并确保有限值
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate)[0]
            spectral_bandwidth = np.nan_to_num(spectral_bandwidth, nan=0.0, posinf=self.sample_rate/4)
            
            # 计算频谱熵并处理边界情况
            entropy_values = []
            for frame in mel_spec_db.T:
                if np.any(np.isfinite(frame)):
                    frame = frame[np.isfinite(frame)]
                    if len(frame) > 0:
                        # 归一化并确保非负
                        frame = frame - np.min(frame)
                        if np.sum(frame) > 0:
                            frame = frame / np.sum(frame)
                            entropy_values.append(entropy(frame))
                        else:
                            entropy_values.append(0.0)
                    else:
                        entropy_values.append(0.0)
                else:
                    entropy_values.append(0.0)
            
            spectral_entropy = np.mean(entropy_values) if entropy_values else 0.0
            
            # 归一化各个特征，确保在0-100范围内
            mean_centroid = np.mean(spectral_centroids)
            mean_bandwidth = np.mean(spectral_bandwidth)
            
            centroid_score = np.clip(100 * (1 - mean_centroid / (self.sample_rate/2)), 0, 100)
            bandwidth_score = np.clip(100 * (1 - mean_bandwidth / (self.sample_rate/4)), 0, 100)
            
            # 确保熵得分在0-100范围内
            max_entropy = np.log2(mel_spec.shape[0])
            entropy_score = np.clip(100 * (1 - spectral_entropy / max_entropy), 0, 100)
            
            # 组合得分
            spectral_score = np.clip(
                0.4 * centroid_score + 
                0.3 * bandwidth_score + 
                0.3 * entropy_score,
                0, 100
            )
            
            return float(spectral_score)
            
        except Exception as e:
            print(f"Error in spectral features calculation: {str(e)}")
            return 50.0  # 返回中间值作为默认值

    def calculate_healing_score(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Calculate individual scores
            roughness_score = self.calculate_roughness(y)
            consonance_score = self.calculate_consonance(y)
            spectral_score = self.calculate_spectral_features(y)
            
            # 使用新的权重
            weights = {
                'roughness': 0.4,
                'consonance': 0.3,
                'spectral': 0.3
            }
            
            # Calculate weighted average
            total_score = (
                weights['roughness'] * roughness_score +
                weights['consonance'] * consonance_score +
                weights['spectral'] * spectral_score
            )
            
            # Apply sigmoid transformation
            transformed_score = sigmoid_transform(total_score)
            
            return {
                'total_score': transformed_score,
                'roughness_score': roughness_score,
                'consonance_score': consonance_score,
                'spectral_score': spectral_score
            }
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return None 