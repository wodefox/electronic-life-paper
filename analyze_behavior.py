import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import os
import glob

class BehaviorAnalyzer:
    def __init__(self, csv_files=None):
        """Initialize analyzer with CSV files"""
        self.csv_files = csv_files if csv_files else glob.glob('autonomous_life_output_seed_*.csv')
        self.data_frames = {}
        self.action_map = {'rest': 0, 'explore': 1}
        self.action_labels = {0: 'rest', 1: 'explore'}
        
        # Load data
        for file in self.csv_files:
            seed = file.split('seed_')[1].split('.')[0]
            self.data_frames[seed] = pd.read_csv(file)
    
    def load_data(self, file_path):
        """Load single CSV file"""
        return pd.read_csv(file_path)
    
    def analyze_single_experiment(self, df, seed=None):
        """Analyze a single experiment"""
        print(f"=== 行为序列分析 (Seed: {seed}) ===")
        
        # 提取行为序列
        actions = df['action'].tolist()
        print(f"总步数: {len(actions)}")
        
        # 统计连续相同行为的长度
        sequences = self._get_action_sequences(actions)
        print(f"总序列数: {len(sequences)}")
        
        # 计算交替频率
        alternations = len(sequences) - 1
        print(f"交替次数: {alternations}")
        print(f"平均每步交替概率: {alternations / len(actions):.4f}")
        
        # 统计explore和rest的总时长
        explore_total = sum(length for action, length in sequences if action == 'explore')
        rest_total = sum(length for action, length in sequences if action == 'rest')
        
        print(f"\nexplore总时长: {explore_total} 步 ({explore_total / len(actions) * 100:.2f}%)")
        print(f"rest总时长: {rest_total} 步 ({rest_total / len(actions) * 100:.2f}%)")
        
        # 分析explore序列长度
        explore_sequences = [length for action, length in sequences if action == 'explore']
        print(f"\nexplore序列分析:")
        print(f"  次数: {len(explore_sequences)}")
        print(f"  平均长度: {sum(explore_sequences) / len(explore_sequences):.2f} 步")
        print(f"  最长: {max(explore_sequences)} 步")
        print(f"  最短: {min(explore_sequences)} 步")
        
        # 分析rest序列长度
        rest_sequences = [length for action, length in sequences if action == 'rest']
        print(f"\nrest序列分析:")
        print(f"  次数: {len(rest_sequences)}")
        print(f"  平均长度: {sum(rest_sequences) / len(rest_sequences):.2f} 步")
        print(f"  最长: {max(rest_sequences)} 步")
        print(f"  最短: {min(rest_sequences)} 步")
        
        # 查看前20个序列模式
        print(f"\n前20个序列模式:")
        for i, (action, length) in enumerate(sequences[:20]):
            print(f"  {i+1}. {action} × {length}")
        
        # 检查是否有明显的周期性
        print(f"\n=== 周期性分析 ===")
        self._analyze_periodicity(actions, seed)
        
        # 计算转移概率矩阵
        print(f"\n=== 转移概率矩阵 ===")
        self._calculate_transition_matrix(actions, seed)
        
        # 自相关分析
        print(f"\n=== 自相关分析 ===")
        self._analyze_autocorrelation(actions, seed)
        
        # 平均归一化统计
        print(f"\n=== 平均归一化统计 ===")
        self._calculate_normalized_stats(df, actions, seed)
        
        # 事件响应分析
        print(f"\n=== 事件响应分析 ===")
        self._analyze_event_response(df, seed)
        
        return {
            'sequences': sequences,
            'explore_total': explore_total,
            'rest_total': rest_total,
            'alternations': alternations
        }
    
    def _get_action_sequences(self, actions):
        """Get sequences of consecutive actions"""
        if not actions:
            return []
        
        sequences = []
        current_action = actions[0]
        current_length = 1
        
        for action in actions[1:]:
            if action == current_action:
                current_length += 1
            else:
                sequences.append((current_action, current_length))
                current_action = action
                current_length = 1
        # 添加最后一个序列
        sequences.append((current_action, current_length))
        
        return sequences
    
    def _analyze_periodicity(self, actions, seed=None):
        """Analyze periodicity using FFT"""
        # Convert actions to binary sequence
        binary_actions = [self.action_map[action] for action in actions]
        
        # Apply FFT
        fft_result = np.fft.fft(binary_actions)
        fft_freq = np.fft.fftfreq(len(binary_actions))
        fft_magnitude = np.abs(fft_result)
        
        # Find peaks in FFT magnitude
        positive_freqs = fft_freq > 0
        peaks, _ = find_peaks(fft_magnitude[positive_freqs], height=0.1*np.max(fft_magnitude[positive_freqs]))
        
        # Get dominant frequencies
        dominant_freqs = fft_freq[positive_freqs][peaks]
        dominant_mags = fft_magnitude[positive_freqs][peaks]
        
        # Sort by magnitude
        sorted_indices = np.argsort(dominant_mags)[::-1]
        dominant_freqs = dominant_freqs[sorted_indices]
        dominant_mags = dominant_mags[sorted_indices]
        
        print("FFT 分析结果:")
        for freq, mag in zip(dominant_freqs[:3], dominant_mags[:3]):
            if freq > 0:
                period = 1 / freq
                print(f"  频率: {freq:.4f}, 周期: {period:.2f} 步, 振幅: {mag:.2f}")
        
        # 可视化FFT结果
        plt.figure(figsize=(10, 6))
        plt.plot(fft_freq[positive_freqs], fft_magnitude[positive_freqs])
        plt.title(f'Action Sequence FFT (Seed: {seed})')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.savefig(f'fft_analysis_seed_{seed}.png' if seed else 'fft_analysis.png')
        print(f"  FFT 图已保存: fft_analysis_seed_{seed}.png" if seed else "  FFT 图已保存: fft_analysis.png")
        plt.close()
    
    def _calculate_transition_matrix(self, actions, seed=None):
        """Calculate and visualize Markov transition matrix"""
        # Initialize transition matrix
        n_actions = len(self.action_map)
        transition_matrix = np.zeros((n_actions, n_actions))
        
        # Count transitions
        for i in range(len(actions) - 1):
            from_action = self.action_map[actions[i]]
            to_action = self.action_map[actions[i+1]]
            transition_matrix[from_action, to_action] += 1
        
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_prob = transition_matrix / row_sums[:, np.newaxis] if row_sums.any() else transition_matrix
        
        # Print transition matrix
        print("转移概率矩阵:")
        print("          to")
        print("          rest    explore")
        for from_action in range(n_actions):
            from_label = self.action_labels[from_action]
            rest_prob = transition_prob[from_action, 0] if row_sums[from_action] > 0 else 0.0
            explore_prob = transition_prob[from_action, 1] if row_sums[from_action] > 0 else 0.0
            print(f"from {from_label:<8} {rest_prob:.4f}    {explore_prob:.4f}")
        
        # Visualize transition matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(transition_prob, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Transition Probability')
        plt.xticks(range(n_actions), [self.action_labels[i] for i in range(n_actions)])
        plt.yticks(range(n_actions), [self.action_labels[i] for i in range(n_actions)])
        plt.title(f'Transition Probability Matrix (Seed: {seed})')
        plt.xlabel('To Action')
        plt.ylabel('From Action')
        
        # Add probability values to the plot
        for i in range(n_actions):
            for j in range(n_actions):
                if row_sums[i] > 0:
                    plt.text(j, i, f'{transition_prob[i, j]:.3f}', ha='center', va='center', color='white')
        
        plt.tight_layout()
        plt.savefig(f'transition_matrix_seed_{seed}.png' if seed else 'transition_matrix.png')
        print(f"  转移矩阵图已保存: transition_matrix_seed_{seed}.png" if seed else "  转移矩阵图已保存: transition_matrix.png")
        plt.close()
    
    def _analyze_autocorrelation(self, actions, seed=None):
        """Analyze autocorrelation"""
        # Convert actions to binary sequence
        binary_actions = [self.action_map[action] for action in actions]
        
        # Calculate autocorrelation
        n = len(binary_actions)
        autocorr = np.correlate(binary_actions, binary_actions, mode='full') / n
        autocorr = autocorr[n-1:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Find significant lags
        significant_lags = np.where(np.abs(autocorr) > 2 / np.sqrt(n))[0]
        significant_lags = significant_lags[significant_lags > 0]  # Exclude lag 0
        
        print("自相关分析结果:")
        print(f"  显著滞后: {significant_lags[:5]}" if len(significant_lags) > 0 else "  无显著滞后")
        
        # Visualize autocorrelation
        plt.figure(figsize=(10, 6))
        plt.plot(autocorr)
        plt.axhline(y=2/np.sqrt(n), color='r', linestyle='--', label=f'95% 置信区间')
        plt.axhline(y=-2/np.sqrt(n), color='r', linestyle='--')
        plt.title(f'Action Sequence Autocorrelation (Seed: {seed})')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'autocorrelation_seed_{seed}.png' if seed else 'autocorrelation.png')
        print(f"  自相关图已保存: autocorrelation_seed_{seed}.png" if seed else "  自相关图已保存: autocorrelation.png")
        plt.close()
    
    def _calculate_normalized_stats(self, df, actions, seed=None):
        """Calculate normalized statistics"""
        # Resource find rate per unit time
        total_finds = df['found_resource'].sum()
        find_rate = total_finds / len(actions) if len(actions) > 0 else 0
        print(f"资源发现率: {find_rate:.4f} 每步")
        
        # Average energy change per explore/rest
        explore_rows = df[df['action'] == 'explore']
        rest_rows = df[df['action'] == 'rest']
        
        if not explore_rows.empty:
            avg_energy_explore = explore_rows['energy'].diff().mean()
            print(f"explore 平均能量变化: {avg_energy_explore:.4f}")
        
        if not rest_rows.empty:
            avg_energy_rest = rest_rows['energy'].diff().mean()
            print(f"rest 平均能量变化: {avg_energy_rest:.4f}")
        
        # Average integrity change per action type
        if not explore_rows.empty:
            avg_integrity_explore = explore_rows['integrity'].diff().mean()
            print(f"explore 平均完整性变化: {avg_integrity_explore:.4f}")
        
        if not rest_rows.empty:
            avg_integrity_rest = rest_rows['integrity'].diff().mean()
            print(f"rest 平均完整性变化: {avg_integrity_rest:.4f}")
    
    def _analyze_event_response(self, df, seed=None):
        """Analyze event-triggered responses"""
        # Find resource find events
        resource_events = df[df['found_resource'] == True].index.tolist()
        print(f"资源发现事件数: {len(resource_events)}")
        
        if len(resource_events) < 5:
            print("  事件数不足，跳过事件响应分析")
            return
        
        # Define window size (steps before and after event)
        window_size = 20
        
        # Initialize arrays to store windowed data
        energy_windows = []
        integrity_windows = []
        drive_windows = []
        
        # Extract windows around each event
        for event_idx in resource_events:
            start_idx = max(0, event_idx - window_size)
            end_idx = min(len(df), event_idx + window_size + 1)
            
            # Skip if window is too small
            if end_idx - start_idx < window_size * 2 + 1:
                continue
            
            # Extract windowed data
            window_df = df.iloc[start_idx:end_idx]
            
            # Normalize time around event
            time = np.arange(-window_size, window_size + 1)
            
            # Store data
            energy_windows.append(window_df['energy'].values)
            integrity_windows.append(window_df['integrity'].values)
            drive_windows.append(window_df['drive'].values)
        
        # Calculate average response
        if energy_windows:
            avg_energy_response = np.mean(energy_windows, axis=0)
            avg_integrity_response = np.mean(integrity_windows, axis=0)
            avg_drive_response = np.mean(drive_windows, axis=0)
            
            # Calculate confidence intervals (95%)
            energy_std = np.std(energy_windows, axis=0) / np.sqrt(len(energy_windows))
            integrity_std = np.std(integrity_windows, axis=0) / np.sqrt(len(integrity_windows))
            drive_std = np.std(drive_windows, axis=0) / np.sqrt(len(drive_windows))
            
            # Plot event responses
            time = np.arange(-window_size, window_size + 1)
            
            plt.figure(figsize=(12, 8))
            
            # Energy response
            plt.subplot(3, 1, 1)
            plt.plot(time, avg_energy_response, label='Average Energy')
            plt.fill_between(time, avg_energy_response - 1.96*energy_std, avg_energy_response + 1.96*energy_std, alpha=0.3)
            plt.axvline(x=0, color='r', linestyle='--', label='Resource Found')
            plt.title(f'Event-Triggered Energy Response (Seed: {seed})')
            plt.ylabel('Energy')
            plt.grid(True)
            plt.legend()
            
            # Integrity response
            plt.subplot(3, 1, 2)
            plt.plot(time, avg_integrity_response, label='Average Integrity')
            plt.fill_between(time, avg_integrity_response - 1.96*integrity_std, avg_integrity_response + 1.96*integrity_std, alpha=0.3)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Event-Triggered Integrity Response')
            plt.ylabel('Integrity')
            plt.grid(True)
            plt.legend()
            
            # Drive response
            plt.subplot(3, 1, 3)
            plt.plot(time, avg_drive_response, label='Average Drive')
            plt.fill_between(time, avg_drive_response - 1.96*drive_std, avg_drive_response + 1.96*drive_std, alpha=0.3)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Event-Triggered Drive Response')
            plt.xlabel('Steps Relative to Resource Find')
            plt.ylabel('Drive')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'event_response_seed_{seed}.png' if seed else 'event_response.png')
            print(f"  事件响应图已保存: event_response_seed_{seed}.png" if seed else "  事件响应图已保存: event_response.png")
            plt.close()
    
    def compare_multiple_experiments(self):
        """Compare multiple experiments"""
        if len(self.data_frames) < 2:
            print("=== 批量试验对比 ===")
            print("  可用试验数不足，跳过对比分析")
            return
        
        print(f"=== 批量试验对比 (共 {len(self.data_frames)} 个试验) ===")
        
        # Collect statistics from all experiments
        stats_data = {
            'seed': [],
            'total_steps': [],
            'explore_total': [],
            'rest_total': [],
            'alternations': [],
            'explore_ratio': [],
            'alternation_ratio': []
        }
        
        for seed, df in self.data_frames.items():
            actions = df['action'].tolist()
            sequences = self._get_action_sequences(actions)
            explore_total = sum(length for action, length in sequences if action == 'explore')
            rest_total = sum(length for action, length in sequences if action == 'rest')
            alternations = len(sequences) - 1
            
            stats_data['seed'].append(seed)
            stats_data['total_steps'].append(len(actions))
            stats_data['explore_total'].append(explore_total)
            stats_data['rest_total'].append(rest_total)
            stats_data['alternations'].append(alternations)
            stats_data['explore_ratio'].append(explore_total / len(actions) if len(actions) > 0 else 0)
            stats_data['alternation_ratio'].append(alternations / len(actions) if len(actions) > 0 else 0)
        
        # Convert to DataFrame for analysis
        stats_df = pd.DataFrame(stats_data)
        
        # Calculate mean and confidence intervals
        print("统计指标均值与95%置信区间:")
        for col in ['explore_ratio', 'alternation_ratio']:
            mean_val = stats_df[col].mean()
            ci_low, ci_high = stats.t.interval(0.95, len(stats_df)-1, loc=mean_val, scale=stats.sem(stats_df[col]))
            print(f"  {col}: {mean_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        
        # Visualize comparison
        plt.figure(figsize=(12, 6))
        
        # Explore ratio comparison
        plt.subplot(1, 2, 1)
        plt.bar(stats_df['seed'], stats_df['explore_ratio'])
        plt.axhline(y=stats_df['explore_ratio'].mean(), color='r', linestyle='--', label=f'Mean: {stats_df["explore_ratio"].mean():.4f}')
        plt.title('Explore Ratio Across Experiments')
        plt.xlabel('Seed')
        plt.ylabel('Explore Ratio')
        plt.ylim(0, 1)
        plt.grid(True, axis='y')
        plt.legend()
        
        # Alternation ratio comparison
        plt.subplot(1, 2, 2)
        plt.bar(stats_df['seed'], stats_df['alternation_ratio'])
        plt.axhline(y=stats_df['alternation_ratio'].mean(), color='r', linestyle='--', label=f'Mean: {stats_df["alternation_ratio"].mean():.4f}')
        plt.title('Alternation Ratio Across Experiments')
        plt.xlabel('Seed')
        plt.ylabel('Alternation Ratio')
        plt.ylim(0, 0.5)
        plt.grid(True, axis='y')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('experiment_comparison.png')
        print("  试验对比图已保存: experiment_comparison.png")
        plt.close()
    
    def run_all_analyses(self):
        """Run all analyses"""
        # Analyze each experiment individually
        for seed, df in self.data_frames.items():
            self.analyze_single_experiment(df, seed)
            print("="*60)
        
        # Compare multiple experiments if available
        self.compare_multiple_experiments()

# Command line interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Autonomous Life System Behavior')
    parser.add_argument('--files', nargs='+', help='CSV files to analyze (default: all seed files)')
    parser.add_argument('--single-file', help='Analyze a single CSV file')
    
    args = parser.parse_args()
    
    if args.single_file:
        # Analyze a single file
        analyzer = BehaviorAnalyzer(csv_files=[args.single_file])
        df = analyzer.load_data(args.single_file)
        analyzer.analyze_single_experiment(df)
    else:
        # Analyze multiple files
        csv_files = args.files if args.files else glob.glob('autonomous_life_output_seed_*.csv')
        if not csv_files:
            print("未找到CSV文件，请确保生成了模拟数据")
            exit(1)
        
        print(f"找到 {len(csv_files)} 个CSV文件，开始分析...")
        analyzer = BehaviorAnalyzer(csv_files=csv_files)
        analyzer.run_all_analyses()
