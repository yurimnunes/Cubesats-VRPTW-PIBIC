import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data (replace with your actual DataFrame)
data = {
    'Instance': range(1,37),
    'T': [96]*12 + [144]*12 + [192]*12,
    'S': [5]*4 + [10]*4 + [15]*4 + [5]*4 + [10]*4 + [15]*4 + [5]*4 + [10]*4 + [15]*4,
    'Js': [8,8,12,12]*9,
    'Jc': [4,8,4,8]*9,
    'Structured_Time': [0.5,0.8,0.9,1.4,0.8,1.2,1.8,2.5,1.1,2.2,2.2,3.4,
                      1.8,1.9,2.7,6.6,3.6,5.3,8.1,8.8,10.4,19.2,11.4,26.3,
                      7.1,11.7,80.6,18.2,22.6,26.6,228.4,133.6,55.4,48.8,138.9,115.9],
    'Naive_Time': [857.2,3600,3600,3060,3600,3600,3600,3600,3600,3600,3600,3600,
                  3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,
                  3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600,3600]
}

df = pd.DataFrame(data)
df['Total_Tasks'] = df['Js'] + df['Jc']

# Create figure with subplots
plt.figure(figsize=(14, 10))
gs = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])

# Main comparison plot
ax1 = plt.subplot(gs[0, :])
metrics = ['Time (s)', 'Iterations', 'Columns']
structured = [29.6, 5.3, 411]
naive = [3429.1, 388.3, 13382]
ratios = [n/s for s,n in zip(structured, naive)]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, structured, width, label='Structured', color='#2ca02c')
bars2 = ax1.bar(x + width/2, naive, width, label='Naive', color='#d62728')

ax1.set_ylabel('Metric Value', fontsize=12)
ax1.set_yscale('log')
ax1.set_title('Performance Comparison: Structured vs. Naive B&P', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()

# Add ratio labels
for i, ratio in enumerate(ratios):
    ax1.text(x[i], max(structured[i], naive[i])*1.2, 
            f'{ratio:.0f}x', ha='center', fontsize=10)

# Scalability plot
ax2 = plt.subplot(gs[1, 0])
ax2.scatter(df['Total_Tasks'], df['Structured_Time'], 
           color='#2ca02c', label='Structured', alpha=0.7)
ax2.scatter(df['Total_Tasks'], df['Naive_Time'], 
           color='#d62728', label='Naive', alpha=0.7)
ax2.set_yscale('log')
ax2.set_xlabel('Total Tasks (Js + Jc)', fontsize=12)
ax2.set_ylabel('Solution Time (s)', fontsize=12)
ax2.set_title('Time Complexity Analysis', fontsize=14)
ax2.legend()
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

# Degeneracy example table
ax3 = plt.subplot(gs[1, 1])
ax3.axis('off')
table_data = [
    ['Metric', 'Structured', 'Naive'],
    ['Time (s)', 0.8, '>3600'],
    ['Iterations', 3, 962],
    ['Columns', 138, '14,262']
]
table = ax3.table(cellText=table_data, 
                 colLabels=None, 
                 cellLoc='center', 
                 loc='center',
                 bbox=[0.1, 0.3, 0.8, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
ax3.set_title('Instance #2: Degeneracy Comparison', fontsize=14, y=0.9)

plt.tight_layout()
plt.savefig('bap_performance_comparison.pdf', bbox_inches='tight')
plt.show()