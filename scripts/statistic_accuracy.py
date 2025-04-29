import os
import json
from collections import defaultdict
import re

def collect_accuracy_data(root_dir):
    # 存储数据结构：model_name -> {dataset -> accuracy}
    accuracy_data = defaultdict(dict)
    n_data = defaultdict(int)

    datasets = set()
    models = set()
    
    # 遍历目录结构
    for root, dirs, files in os.walk(root_dir):
        if 'summary.json' in files:
            # 解析路径信息
            path_parts = root.split(os.sep)
            dataset = path_parts[-2]  # 父目录是数据集名称
            model_dir = path_parts[-1]  # 当前目录是模型目录
            
            # 提取模型名称
            model_name = model_dir.split(dataset)[0]
            
            # 读取summary.json
            with open(os.path.join(root, 'summary.json'), 'r') as f:
                summary = json.load(f)
                accuracy = summary.get('accuracy', None)
                # accuracy = summary.get('exceed_in_total', None)
                
                n = summary['configuration']['sampling_params']['n']
                
                if n > n_data[(model_name, dataset)]:
                    accuracy_data[model_name][dataset] = accuracy
                    n_data[(model_name, dataset)] = n

                datasets.add(dataset)
                models.add(model_name)
    
    return accuracy_data, sorted(datasets), sorted(models)

def generate_markdown_table(accuracy_data, datasets, models):
    # 准备表格数据
    table = []
    
    # 表头
    header = ["Model"] + datasets + ["Average"]
    table.append(header)
    
    # 表格分隔线
    table.append(["-"] * len(header))
    
    # 每行数据
    for model in models:
        row = [model]
        total = 0
        count = 0
        
        for dataset in datasets:
            accuracy = accuracy_data[model].get(dataset, "-")
            if isinstance(accuracy, (int, float)):
                row.append(f"{accuracy * 100:.1f}")
                total += accuracy
                count += 1
            else:
                row.append(accuracy)
        
        # 计算平均准确率
        avg = total / count if count > 0 else "-"
        row.append(f"{avg * 100:.1f}" if isinstance(avg, (int, float)) else avg)
        table.append(row)
    
    # 生成Markdown格式
    markdown = ""
    for row in table:
        markdown += "| " + " | ".join(str(x) for x in row) + " |\n"
    
    return markdown

def main():
    root_dir = "/mnt/workspace/user/sunwangtao/Skythought/skythought/evaluate_results-temp0.6-tp95/math-long-cot-20k"  # 当前目录，可以根据需要修改
    accuracy_data, datasets, models = collect_accuracy_data(root_dir)
    markdown_table = generate_markdown_table(accuracy_data, datasets, models)
    
    # 保存到文件
    with open("model_performance.md", "w") as f:
        f.write("# Model Performance Across Datasets\n\n")
        f.write(markdown_table)
    
    print("Markdown table generated successfully in model_performance.md")

if __name__ == "__main__":
    main()