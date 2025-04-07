import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_contrasting_cases_with_exceed_stats(model1_path, model2_path, max_length=16385):
    # Load both models' results
    with open(model1_path, 'r') as f:
        model1_results = json.load(f)
    with open(model2_path, 'r') as f:
        model2_results = json.load(f)
    
    # Verify both models evaluated the same samples
    common_samples = set(model1_results.keys()) & set(model2_results.keys())
    print(f"Number of common samples: {len(common_samples)}")
    
    # Initialize case collections with exceed tracking
    cases = {
        'model1_correct': {
            'length_diffs': [],
            'model2_exceeded': 0,
            'model2_wrong_content': 0
        },
        'model2_correct': {
            'length_diffs': [],
            'model1_exceeded': 0,
            'model1_wrong_content': 0
        },
        'both_wrong': {
            'count': 0,
            'both_exceeded': 0,
            'model1_exceeded_only': 0,
            'model2_exceeded_only': 0,
            'neither_exceeded': 0
        },
        'exceed_stats': {
            'model1_total_exceeded': 0,
            'model2_total_exceeded': 0,
            'both_exceeded': 0
        }
    }
    
    for sample_id in common_samples:
        # Get first predictions
        m1_resp = model1_results[sample_id]['responses'][0]
        m2_resp = model2_results[sample_id]['responses'][0]
        
        m1_length = (model1_results[sample_id]['token_usages'][0]['prompt_tokens'] + 
                    model1_results[sample_id]['token_usages'][0]['completion_tokens'])
        m2_length = (model2_results[sample_id]['token_usages'][0]['prompt_tokens'] + 
                    model2_results[sample_id]['token_usages'][0]['completion_tokens'])
        
        # Determine correctness with length limit
        m1_exceeded = m1_length >= max_length
        m2_exceeded = m2_length >= max_length
        
        # Track overall exceed statistics
        if m1_exceeded:
            cases['exceed_stats']['model1_total_exceeded'] += 1
        if m2_exceeded:
            cases['exceed_stats']['model2_total_exceeded'] += 1
        if m1_exceeded and m2_exceeded:
            cases['exceed_stats']['both_exceeded'] += 1
        
        m1_correct = m1_resp['correctness'] and not m1_exceeded
        m2_correct = m2_resp['correctness'] and not m2_exceeded
        
        length_diff = m1_length - m2_length
        
        # Classify cases
        if m1_correct and not m2_correct:
            cases['model1_correct']['length_diffs'].append(length_diff)
            if m2_exceeded:
                cases['model1_correct']['model2_exceeded'] += 1
            else:
                cases['model1_correct']['model2_wrong_content'] += 1
                
        elif m2_correct and not m1_correct:
            cases['model2_correct']['length_diffs'].append(length_diff)
            if m1_exceeded:
                cases['model2_correct']['model1_exceeded'] += 1
            else:
                cases['model2_correct']['model1_wrong_content'] += 1
                
        else:  # Both wrong
            cases['both_wrong']['count'] += 1
            if m1_exceeded and m2_exceeded:
                cases['both_wrong']['both_exceeded'] += 1
            elif m1_exceeded:
                cases['both_wrong']['model1_exceeded_only'] += 1
            elif m2_exceeded:
                cases['both_wrong']['model2_exceeded_only'] += 1
            else:
                cases['both_wrong']['neither_exceeded'] += 1
    
    # Calculate statistics
    stats = {
        'model1_correct': {
            'count': len(cases['model1_correct']['length_diffs']),
            'model2_exceeded': cases['model1_correct']['model2_exceeded'],
            'model2_wrong_content': cases['model1_correct']['model2_wrong_content'],
            'avg_length_diff': np.mean(cases['model1_correct']['length_diffs']) if cases['model1_correct']['length_diffs'] else 0,
            'median_length_diff': np.median(cases['model1_correct']['length_diffs']) if cases['model1_correct']['length_diffs'] else 0
        },
        'model2_correct': {
            'count': len(cases['model2_correct']['length_diffs']),
            'model1_exceeded': cases['model2_correct']['model1_exceeded'],
            'model1_wrong_content': cases['model2_correct']['model1_wrong_content'],
            'avg_length_diff': np.mean(cases['model2_correct']['length_diffs']) if cases['model2_correct']['length_diffs'] else 0,
            'median_length_diff': np.median(cases['model2_correct']['length_diffs']) if cases['model2_correct']['length_diffs'] else 0
        },
        'both_wrong': cases['both_wrong'],
        'exceed_stats': cases['exceed_stats'],
        'total_samples': len(common_samples)
    }
    
    # Print results
    print("\n=== Detailed Contrasting Cases Analysis ===")
    print(f"Max length considered: {max_length} tokens")
    print(f"\nTotal samples: {stats['total_samples']}")
    
    print("\nModel1 Correct, Model2 Wrong:")
    print(f"  Total cases: {stats['model1_correct']['count']} ({stats['model1_correct']['count']/stats['total_samples']:.1%})")
    print(f"  - Model2 exceeded length: {stats['model1_correct']['model2_exceeded']} ({stats['model1_correct']['model2_exceeded']/stats['model1_correct']['count']:.1%})")
    print(f"  - Model2 wrong content: {stats['model1_correct']['model2_wrong_content']} ({stats['model1_correct']['model2_wrong_content']/stats['model1_correct']['count']:.1%})")
    print(f"  Average length diff (M1-M2): {stats['model1_correct']['avg_length_diff']:.1f} tokens")
    print(f"  Median length diff: {stats['model1_correct']['median_length_diff']:.1f} tokens")
    
    print("\nModel2 Correct, Model1 Wrong:")
    print(f"  Total cases: {stats['model2_correct']['count']} ({stats['model2_correct']['count']/stats['total_samples']:.1%})")
    print(f"  - Model1 exceeded length: {stats['model2_correct']['model1_exceeded']} ({stats['model2_correct']['model1_exceeded']/stats['model2_correct']['count']:.1%})")
    print(f"  - Model1 wrong content: {stats['model2_correct']['model1_wrong_content']} ({stats['model2_correct']['model1_wrong_content']/stats['model2_correct']['count']:.1%})")
    print(f"  Average length diff (M1-M2): {stats['model2_correct']['avg_length_diff']:.1f} tokens")
    print(f"  Median length diff: {stats['model2_correct']['median_length_diff']:.1f} tokens")
    
    print("\nBoth Wrong:")
    print(f"  Total cases: {stats['both_wrong']['count']} ({stats['both_wrong']['count']/stats['total_samples']:.1%})")
    print(f"  - Both exceeded: {stats['both_wrong']['both_exceeded']} ({stats['both_wrong']['both_exceeded']/stats['both_wrong']['count']:.1%})")
    print(f"  - Only Model1 exceeded: {stats['both_wrong']['model1_exceeded_only']} ({stats['both_wrong']['model1_exceeded_only']/stats['both_wrong']['count']:.1%})")
    print(f"  - Only Model2 exceeded: {stats['both_wrong']['model2_exceeded_only']} ({stats['both_wrong']['model2_exceeded_only']/stats['both_wrong']['count']:.1%})")
    print(f"  - Neither exceeded: {stats['both_wrong']['neither_exceeded']} ({stats['both_wrong']['neither_exceeded']/stats['both_wrong']['count']:.1%})")
    
    print("\nOverall Exceed Statistics:")
    print(f"  Model1 total exceeded: {stats['exceed_stats']['model1_total_exceeded']} ({stats['exceed_stats']['model1_total_exceeded']/stats['total_samples']:.1%})")
    print(f"  Model2 total exceeded: {stats['exceed_stats']['model2_total_exceeded']} ({stats['exceed_stats']['model2_total_exceeded']/stats['total_samples']:.1%})")
    print(f"  Both exceeded: {stats['exceed_stats']['both_exceeded']} ({stats['exceed_stats']['both_exceeded']/stats['total_samples']:.1%})")
    
    # Visualization
    if stats['model1_correct']['count'] > 0 or stats['model2_correct']['count'] > 0:
        plt.figure(figsize=(16, 6))
        
        # Case distribution pie chart
        plt.subplot(1, 3, 1)
        case_counts = [
            stats['model1_correct']['count'],
            stats['model2_correct']['count'],
            stats['both_wrong']['count']
        ]
        case_labels = [
            'Model1✓\nModel2✗\n(n={})'.format(stats['model1_correct']['count']),
            'Model2✓\nModel1✗\n(n={})'.format(stats['model2_correct']['count']),
            'Both✗\n(n={})'.format(stats['both_wrong']['count'])
        ]
        plt.pie(case_counts, labels=case_labels, autopct='%1.1f%%',
               colors=['lightgreen', 'lightcoral', 'lightgray'])
        plt.title('Case Distribution')
        
        # Wrong model reasons (Model1 correct cases)
        plt.subplot(1, 3, 2)
        if stats['model1_correct']['count'] > 0:
            reasons = [
                stats['model1_correct']['model2_exceeded'],
                stats['model1_correct']['model2_wrong_content']
            ]
            reason_labels = [
                'Exceeded\nlength\n(n={})'.format(stats['model1_correct']['model2_exceeded']),
                'Wrong\ncontent\n(n={})'.format(stats['model1_correct']['model2_wrong_content'])
            ]
            plt.pie(reasons, labels=reason_labels, autopct='%1.1f%%',
                   colors=['orange', 'gold'])
            plt.title('When Model1✓ Model2✗:\nWhy Model2 Failed')
        
        # Wrong model reasons (Model2 correct cases)
        plt.subplot(1, 3, 3)
        if stats['model2_correct']['count'] > 0:
            reasons = [
                stats['model2_correct']['model1_exceeded'],
                stats['model2_correct']['model1_wrong_content']
            ]
            reason_labels = [
                'Exceeded\nlength\n(n={})'.format(stats['model2_correct']['model1_exceeded']),
                'Wrong\ncontent\n(n={})'.format(stats['model2_correct']['model1_wrong_content'])
            ]
            plt.pie(reasons, labels=reason_labels, autopct='%1.1f%%',
                   colors=['blue', 'lightblue'])
            plt.title('When Model2✓ Model1✗:\nWhy Model1 Failed')
        
        plt.tight_layout()
        plt.show()
        
        # Length differences boxplot
        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        if stats['model1_correct']['count'] > 0:
            data.append(cases['model1_correct']['length_diffs'])
            labels.append('Model1✓ Model2✗\n(n={})'.format(stats['model1_correct']['count']))
        if stats['model2_correct']['count'] > 0:
            data.append(cases['model2_correct']['length_diffs'])
            labels.append('Model2✓ Model1✗\n(n={})'.format(stats['model2_correct']['count']))
        
        box = plt.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.ylabel('Length Difference (Model1 - Model2) in tokens')
        plt.title('Length Differences in Contrasting Cases')
        plt.axhline(0, color='gray', linestyle='--')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    return stats

# Example usage
model1_path = 'skythought/evaluate_results/Bespoke-Stratos-17k/math500/saves_Bespoke-Stratos-17k_Qwen2.5-7B-Instruct_full_math500_408c01/results.json'
model2_path = 'skythought/evaluate_results/Bespoke-Stratos-17k/math500/saves_Bespoke-Stratos-17k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_math500_d19ecb/results.json'

stats = analyze_contrasting_cases_with_exceed_stats(model1_path, model2_path)