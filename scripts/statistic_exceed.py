import os
import json
from tqdm import tqdm

ROOT_DIR = "skythought/evaluate_results-temp0.6-tp95/math-long-cot-20k"  # 替换为你的根目录
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


if __name__ == "__main__":
    for root, dirs, files in os.walk(ROOT_DIR):
        for d in dirs:
            folder_path = os.path.join(root, d)
            result = process_folder(folder_path)

    # print("处理完成，以下是结果统计前几项：")
    # for s in stats[:5]:
    #     print(s)
