import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import argparse
from roughness_analyzer import RoughnessAnalyzer

warnings.filterwarnings('ignore')
plt.style.use('seaborn')  # 使用更美观的绘图风格

# 添加全局变量用于控制程序终止
should_exit = False

def signal_handler(signum, frame):
    """处理信号的函数"""
    global should_exit
    if signum == signal.SIGINT:  # Ctrl+C
        print("\n接收到 Ctrl+C，正在安全退出...")
    elif signum == signal.SIGTERM:  # 终止信号
        print("\n接收到终止信号，正在安全退出...")
    
    should_exit = True
    sys.exit(0)

# 注册信号处理器
def setup_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    if sys.platform != 'win32':  # 在非Windows系统上
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

class MusicAnalyzer:
    def __init__(self, base_path=None):
        # 允许通过参数传入路径，否则使用默认路径
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.healing_path = self.base_path / "healing_music"  # 改用英文路径名
        self.non_healing_path = self.base_path / "non_healing_music"
        
        # 检查必要的目录是否存在
        if not self.healing_path.exists():
            self.healing_path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {self.healing_path}")
        
        if not self.non_healing_path.exists():
            self.non_healing_path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {self.non_healing_path}")
        
        self.roughness_analyzer = RoughnessAnalyzer()  # 直接初始化粗糙度分析器
        
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
    
    def extract_pitch(self, audio_path):
        """提取音频的音高序列（改进错误处理）"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, duration=30)
            y = librosa.effects.preemphasis(y)
            
            # 设置更保守的参数
            hop_length = 512
            fmin = librosa.note_to_hz('C2')
            fmax = librosa.note_to_hz('C7')
            
            # 使用更稳定的音高检测方法
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, 
                hop_length=hop_length,
                fmin=fmin, 
                fmax=fmax,
                threshold=0.1  # 添加阈值过滤噪音
            )
            
            # 使用更安全的方式选择音高
            pitch_sequence = np.zeros_like(magnitudes[0])
            for i in range(magnitudes.shape[1]):
                max_index = magnitudes[:, i].argmax()
                if magnitudes[max_index, i] > 0:  # 只选择有足够能量的音高
                    pitch_sequence[i] = pitches[max_index, i]
            
            return pitch_sequence
            
        except Exception as e:
            print(f"音高提取出错: {str(e)}")
            # 返回一个默认的音高序列
            return np.zeros(1024)
    
    def calculate_intervals(self, pitch_sequence):
        """计算音程序列（改进的错误处理）"""
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
    
    def calculate_affinity_score(self, intervals):
        """计算音程亲和力分数"""
        if len(intervals) == 0:
            return 0
        
        total_weight = 0
        for interval in intervals:
            interval = int(round(interval))
            weight = self.interval_weights.get(interval, 0)
            total_weight += weight
            
        avg_weight = total_weight / len(intervals)
        min_score = -0.6
        max_score = 1.0
        
        normalized_score = (avg_weight - min_score) / (max_score - min_score)
        return max(0, min(1, normalized_score))
    
    def analyze_file(self, file_path):
        """分析单个音频文件（改进的错误处理）"""
        if should_exit:
            return None
        try:
            # 音程分析
            pitch_sequence = self.extract_pitch(file_path)
            intervals = self.calculate_intervals(pitch_sequence)
            
            # 确保至少有一些有效的音程
            if len(intervals) > 0:
                interval_score = self.calculate_affinity_score(intervals)
            else:
                interval_score = 0.5  # 设置默认值
            
            # 粗糙度分析
            try:
                roughness_score = self.roughness_analyzer.calculate_roughness(file_path)
            except Exception as e:
                print(f"粗糙度分析出错: {str(e)}")
                roughness_score = 0.5  # 设置默认值
            
            # 综合得分
            final_score = 0.7 * interval_score + 0.3 * roughness_score
            
            return {
                'filename': os.path.basename(file_path),
                'interval_score': interval_score,
                'roughness_score': roughness_score,
                'final_score': final_score
            }
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            # 返回默认结果而不是None
            return {
                'filename': os.path.basename(file_path),
                'interval_score': 0.5,
                'roughness_score': 0.5,
                'final_score': 0.5
            }
    
    def analyze_directory(self, directory):
        """分析目录中的所有音频文件"""
        results = []
        # 支持更多音频格式
        audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.ogg')
        files = [f for f in os.listdir(directory) if f.lower().endswith(audio_extensions)]
        
        if not files:
            print(f"警告: {directory} 中没有找到支持的音频文件")
            return results
        
        print(f"\n正在分析 {directory} 中的音频文件...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for file in files:
                if should_exit:
                    print("\n检测到终止信号，正在保存已完成的分析结果...")
                    break
                file_path = os.path.join(directory, file)
                futures.append(executor.submit(self.analyze_file, file_path))
            
            for future in tqdm(futures, total=len(files)):
                if should_exit:
                    break
                result = future.result()
                if result is not None:
                    results.append(result)
                    
                    # 定期保存中间结果
                    if len(results) % 5 == 0:  # 每分析5个文件保存一次
                        self.save_intermediate_results(results)
        
        return results
    
    def save_intermediate_results(self, results):
        """保存中间结果"""
        try:
            temp_df = pd.DataFrame(results)
            temp_df.to_excel('temp_results.xlsx', index=False)
        except Exception as e:
            print(f"保存中间结果时出错: {str(e)}")
    
    def run_analysis(self):
        """运行完整分析"""
        healing_results = self.analyze_directory(self.healing_path)
        healing_df = pd.DataFrame(healing_results)
        healing_df['category'] = '治愈音乐'
        
        non_healing_results = self.analyze_directory(self.non_healing_path)
        non_healing_df = pd.DataFrame(non_healing_results)
        non_healing_df['category'] = '非治愈音乐'
        
        results_df = pd.concat([healing_df, non_healing_df])
        
        # 确保使用正确的列名进行统计
        score_column = 'final_score'  # 使用final_score作为评分列
        
        stats = {
            '治愈音乐平均分': healing_df[score_column].mean(),
            '非治愈音乐平均分': non_healing_df[score_column].mean(),
            '治愈音乐标准差': healing_df[score_column].std(),
            '非治愈音乐标准差': non_healing_df[score_column].std()
        }
        
        return results_df, stats

def create_analysis_report(results_df, stats_dict, output_path):
    """创建详细的分析报告（添加粗糙度可视化）"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 数据清理：移除NaN值
    results_df = results_df.dropna(subset=['final_score', 'interval_score', 'roughness_score'])
    
    # 创建两个图表：一个用于综合得分，一个用于粗糙度得分
    # 1. 综合得分图表
    plt.figure(figsize=(15, 6))
    healing_scores = results_df[results_df['category'] == '治愈音乐']['final_score']
    non_healing_scores = results_df[results_df['category'] == '非治愈音乐']['final_score']
    
    # 箱线图
    plt.subplot(121)
    box_plot = plt.boxplot([healing_scores, non_healing_scores], 
                         labels=['治愈音乐', '非治愈音乐'],
                         patch_artist=True)
    
    colors = ['lightgreen', 'lightblue']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('音程亲和力分数分布箱线图', fontsize=12, pad=15)
    plt.ylabel('亲和力分数', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 直方图
    plt.subplot(122)
    # 确保数据有效
    if not healing_scores.empty and not non_healing_scores.empty:
        # 计算合适的bins范围
        min_val = min(healing_scores.min(), non_healing_scores.min())
        max_val = max(healing_scores.max(), non_healing_scores.max())
        bins = np.linspace(min_val, max_val, 16)  # 15个区间需要16个边界点
        
        plt.hist(healing_scores, bins=bins, alpha=0.6, label='治愈音乐', color='lightgreen')
        plt.hist(non_healing_scores, bins=bins, alpha=0.6, label='非治愈音乐', color='lightblue')
        plt.title('音程亲和力分数分布直方图', fontsize=12, pad=15)
        plt.xlabel('亲和力分数', fontsize=10)
        plt.ylabel('频数', fontsize=10)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 创建粗糙度得分的可视化
    plt.figure(figsize=(15, 6))
    
    # 粗糙度箱线图
    plt.subplot(121)
    healing_roughness = results_df[results_df['category'] == '治愈音乐']['roughness_score']
    non_healing_roughness = results_df[results_df['category'] == '非治愈音乐']['roughness_score']
    
    box_plot = plt.boxplot([healing_roughness, non_healing_roughness],
                          labels=['治愈音乐', '非治愈音乐'],
                          patch_artist=True)
    
    colors = ['lightgreen', 'lightblue']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('音乐粗糙度得分分布箱线图', fontsize=12, pad=15)
    plt.ylabel('粗糙度得分', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 粗糙度直方图
    plt.subplot(122)
    if not healing_roughness.empty and not non_healing_roughness.empty:
        min_val = min(healing_roughness.min(), non_healing_roughness.min())
        max_val = max(healing_roughness.max(), non_healing_roughness.max())
        bins = np.linspace(min_val, max_val, 16)
        
        plt.hist(healing_roughness, bins=bins, alpha=0.6, label='音乐', color='lightgreen')
        plt.hist(non_healing_roughness, bins=bins, alpha=0.6, label='非治愈音乐', color='lightblue')
        plt.title('音乐粗糙度得分分布直方图', fontsize=12, pad=15)
        plt.xlabel('粗糙度得分', fontsize=10)
        plt.ylabel('频数', fontsize=10)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('roughness_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    setup_signal_handlers()
    print("程序已启动，可以使用以下方式终止程序：")
    print("- Ctrl+C: 立即终止")
    print("- Ctrl+Z: 安全终止（等待当前任务完成）")
    
    parser = argparse.ArgumentParser(description='音乐治愈性分析工具')
    parser.add_argument('--base_path', type=str, default=str(Path.cwd()),
                       help='音频文件的基础路径，默认为当前目录')
    args = parser.parse_args()
    
    analyzer = MusicAnalyzer(args.base_path)
    
    try:
        results_df, stats = analyzer.run_analysis()
        
        if not should_exit:
            # 使用 Path 对象处理路径
            output_path = Path(args.base_path) / 'analysis_results.xlsx'
            create_analysis_report(results_df, stats, str(output_path))
            
            print("\n=== 分析结果 ===")
            print("\n1. 综合得分:")
            for key, value in stats.items():
                print(f"{key}: {value:.4f}")
            
            print("\n2. 分项得分:")
            print("音程分析:")
            print(f"治愈音乐平均音程得分: {results_df[results_df['category']=='治愈音乐']['interval_score'].mean():.4f}")
            print(f"非治愈音乐平均音程得分: {results_df[results_df['category']=='非治愈音乐']['interval_score'].mean():.4f}")
            
            print("\n粗糙度分析:")
            print(f"治愈音乐平均粗糙度: {results_df[results_df['category']=='治愈音乐']['roughness_score'].mean():.4f}")
            print(f"非治愈音乐平均粗糙度: {results_df[results_df['category']=='非治愈音乐']['roughness_score'].mean():.4f}")
            
            print(f"\n详细分析结果已保存到: {output_path}")
            print(f"综合得分分布图已保存到: score_distribution.png")
            print(f"粗糙度分布图已保存到: roughness_distribution.png")
        
    except KeyboardInterrupt:
        print("\n程序已被用户终止")
        sys.exit(0)

if __name__ == "__main__":
    main() 