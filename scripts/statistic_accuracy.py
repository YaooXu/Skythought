import os
import json
from collections import defaultdict
import re

TOKEN_LIMIT = 16384

def get_token_length(text):
    # 简单按空格分词统计（你也可以接入 tokenizer）
    return len(text.strip().split())

def process_folder(folder_path):
    results_path = os.path.join(folder_path, "results.json")
    summary_path = os.path.join(folder_path, "summary.json")

    if not (os.path.exists(results_path) and os.path.exists(summary_path)):
        return None
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        if 'exceed_in_total' in summary:
            print(f"{folder_path} 已经处理过，跳过")
            return None
        
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = 0
    exceed = 0
    total_wrong = 0
    exceed_in_wrong = 0

    for key in results:
        prediction = results[key]
        responses = prediction['responses']
        token_usages = prediction['token_usages']
        
        for response, token_usage in zip(responses, token_usages):
            total += 1
            length = token_usage['completion_tokens'] + token_usage['prompt_tokens']
            correctness = response.get('correctness', False)

            if not correctness:
                total_wrong += 1
                if length == 16385:
                    exceed_in_wrong += 1

            if length == 16385:
                exceed += 1

    proportion_all = exceed / total
    proportion_wrong = exceed_in_wrong / total_wrong

    # 加入 summary.json 中
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    summary["exceed_in_total"] = proportion_all
    summary['exceed_in_wrong'] = proportion_wrong
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # return {
    #     "folder": folder_path,
    #     "exceed": exceed,
    #     "total": total,
    #     "ratio": proportion
    # }


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
    root_dir = "/mnt/workspace/user/sunwangtao/Skythought/skythought/evaluate_results-temp0.7-tp95/math-long-cot-20k"  # 当前目录，可以根据需要修改

    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            folder_path = os.path.join(root, d)
            result = process_folder(folder_path)
    
    accuracy_data, datasets, models = collect_accuracy_data(root_dir)
    markdown_table = generate_markdown_table(accuracy_data, datasets, models)
    
    # 保存到文件
    with open("model_performance.md", "w") as f:
        f.write("# Model Performance Across Datasets\n\n")
        f.write(markdown_table)
    
    print("Markdown table generated successfully in model_performance.md")

if __name__ == "__main__":
    main()