import os
import json


def analyze_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        analysis = {}
        for key, values in data.items():
            if isinstance(values, list) and all(isinstance(val, (int, float)) for val in values):
                max_val = max(values)
                min_val = min(values)
                avg_val = sum(values) / len(values)
                analysis[key] = {
                    'max': max_val,
                    'min': min_val,
                    'avg': avg_val
                }
        return analysis
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"无法解析 {file_path} 中的 JSON 数据。")
    return None


def analyze_all_json_files():
    current_dir = os.getcwd()
    json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]
    for json_file in json_files:
        file_path = os.path.join(current_dir, json_file)
        print(f"分析文件: {json_file}")
        analysis = analyze_json_file(file_path)
        if analysis:
            for key, stat in analysis.items():
                print(f"{key}:")
                print(f"  最大值: {stat['max']}")
                print(f"  最小值: {stat['min']}")
                print(f"  平均值: {stat['avg']}")
            print()


if __name__ == "__main__":
    analyze_all_json_files()