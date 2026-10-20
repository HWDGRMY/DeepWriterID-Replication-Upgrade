import os
import random
import glob
import pandas as pd
from collections import defaultdict


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(BASE_DIR, 'data', 'raw')
    output_dir = os.path.join(BASE_DIR, 'data', 'features')
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔍 正在扫描原始轨迹文件: {raw_dir}")

    # 扫描所有 .wptt 文件
    all_files = glob.glob(os.path.join(raw_dir, '**', '*.wptt'), recursive=True)

    # 按作者分组
    writer_pages = defaultdict(list)
    for f in all_files:
        basename = os.path.basename(f)
        writer_id = basename.split('-')[0]
        page = basename.split('.')[0]
        writer_pages[writer_id].append({'file': f, 'page': page})

    # 随机划分（取消固定种子，让每次运行完全随机，更符合真实开源场景）
    records = []
    for writer_id, items in writer_pages.items():
        random.shuffle(items)
        # 第1页作测试，其余作训练
        test_item = items[0]
        test_item['split'] = 'Test'
        test_item['writer_id'] = writer_id
        records.append(test_item)

        for item in items[1:]:
            item['split'] = 'Train'
            item['writer_id'] = writer_id
            records.append(item)

    # 保存 metadata.csv
    df = pd.DataFrame(records)
    df['writer_id'] = df['writer_id'].astype(str)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    df.to_csv(metadata_path, index=False, encoding='utf-8-sig')

    train_writers = df[df['split'] == 'Train']['writer_id'].nunique()
    test_writers = df[df['split'] == 'Test']['writer_id'].nunique()
    train_samples = len(df[df['split'] == 'Train'])
    test_samples = len(df[df['split'] == 'Test'])

    print(f"✅ 预处理完成！")
    print(f"📊 本地训练集作者数: {train_writers}")
    print(f"📊 本地测试集作者数: {test_writers}")
    print(f"📊 训练集样本数: {train_samples}")
    print(f"📊 测试集样本数: {test_samples}")
    print(f"📌 metadata.csv 已保存至: {metadata_path}")


if __name__ == '__main__':
    main()