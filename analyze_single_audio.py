#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import locale
import argparse
from healing_features import HealingFeatures
from concurrent.futures import ThreadPoolExecutor

# 设置控制台输出编码
if sys.platform.startswith('win'):
    # 设置控制台编码为UTF-8
    os.system('chcp 65001')
    # 设置Python环境编码
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    # 设置区域编码
    try:
        locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass

def analyze_single_file(file_path):
    """
    使用与music_analysis.py相同的方法分析单个音频文件
    
    Args:
        file_path (str): 音频文件的路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到音频文件: {file_path}")
            
        # 检查文件格式
        valid_extensions = {'.mp3', '.wav', '.flac', '.m4a'}
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"不支持的音频格式: {file_ext}。支持的格式: {', '.join(valid_extensions)}")
        
        print(f"\n正在分析音频文件: {os.path.basename(file_path)}")
        print("请稍候...\n")
        
        # 创建分析器实例
        analyzer = HealingFeatures()
        
        # 使用与music_analysis.py相同的处理方法
        def process_file(file_path):
            try:
                scores = analyzer.calculate_healing_score(file_path)
                if scores:
                    return {
                        'filename': os.path.basename(file_path),
                        'total_score': float(scores['total_score']),
                        'roughness_score': float(scores['roughness_score']),
                        'consonance_score': float(scores['consonance_score']),
                        'spectral_score': float(scores['spectral_score'])
                    }
            except Exception as e:
                print(f"处理文件时出错: {str(e)}")
                return None
        
        # 获取分析结果
        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_file, file_path)
            results = future.result()
        
        if results is None:
            raise RuntimeError("分析过程出现错误")
            
        # 打印分析结果
        print("\n" + "="*50)
        print(f"音频文件分析报告: {results['filename']}")
        print("="*50)
        
        # 总分
        print(f"\n总体治愈得分: {results['total_score']:.2f}/100")
        print("-"*30)
        
        # 各维度得分详情
        print("\n详细评分:")
        print(f"1. 音程亲和力得分: {results['consonance_score']:.2f}/100")
        print("   - 评估音乐的和声协和度")
        print("   - 较高分数表示音程关系更和谐")
        
        print(f"\n2. 声音粗糙度得分: {results['roughness_score']:.2f}/100")
        print("   - 评估音乐的平滑度")
        print("   - 较高分数表示声音更柔和平顺")
        
        print(f"\n3. 频谱特征得分: {results['spectral_score']:.2f}/100")
        print("   - 评估音色和频率分布")
        print("   - 较高分数表示音色更纯净、频率分布更合理")
        
        # 评分解释
        print("\n得分解释:")
        total_score = results['total_score']
        if total_score >= 80:
            print("该音频具有很强的治愈特性，音乐和谐、平稳、富有感染力")
        elif total_score >= 60:
            print("该音频具有一定的治愈特性，整体表现良好")
        elif total_score >= 40:
            print("该音频的治愈特性一般，可能需要在某些方面进行改进")
        else:
            print("该音频的治愈特性较弱，建议参考详细评分进行针对性改进")
            
        print("\n" + "="*50)
        
        return results
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        return None

def process_path(path):
    """处理输入的文件路径"""
    if not path:
        return None
        
    # 处理路径
    path = path.strip()
    # 移除开头的&和空格
    if path.startswith('&'):
        path = path.lstrip('& ')
    # 移除引号
    path = path.strip('"').strip("'")
    # 处理Windows路径中的反斜杠
    path = path.replace('\\\\', '\\')
    
    # 规范化路径
    path = os.path.normpath(path)
    
    return path if os.path.exists(path) else None

def get_audio_path():
    """交互式获取音频文件路径"""
    try:
        print("\n欢迎使用音乐治愈性分析工具!")
        print("支持的音频格式: MP3, WAV, FLAC, M4A")
        print("\n请输入音频文件的完整路径（可以直接拖拽文件到此窗口）:")
        
        while True:
            try:
                # 获取用户输入
                path = input().strip()
                
                if path.lower() in ('q', 'quit', 'exit'):
                    return None
                
                # 处理路径
                processed_path = process_path(path)
                if processed_path:
                    return processed_path
                
                print("\n找不到指定文件，请检查路径是否正确。")
                print("请重新输入路径，或输入 'q' 退出:")
                print("提示：可以直接将文件拖入窗口，或复制完整的文件路径")
                    
            except Exception as e:
                print(f"\n输入错误: {str(e)}")
                print("请重新输入路径，或输入 'q' 退出:")
    except Exception as e:
        print(f"\n程序出错: {str(e)}")
        return None

def main():
    try:
        # 检查是否通过命令行参数提供了文件路径
        parser = argparse.ArgumentParser(description='分析单个音频文件的治愈特性')
        parser.add_argument('audio_path', nargs='?', type=str, help='音频文件的路径（可选）')
        
        args = parser.parse_args()
        
        # 处理命令行参数中的路径
        if args.audio_path:
            audio_path = process_path(args.audio_path)
            if not audio_path:
                print(f"\n错误: 找不到文件 '{args.audio_path}'")
                return
        else:
            # 如果没有通过命令行提供路径，则交互式获取
            audio_path = get_audio_path()
        
        if audio_path:
            analyze_single_file(audio_path)
        else:
            print("\n程序已退出。")
            
    except Exception as e:
        print(f"\n程序出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 