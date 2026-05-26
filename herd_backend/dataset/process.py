# import numpy as np

# # 加载npy文件
# data = np.load('dataset/herb_features_.npy')
# print(data)
# data1 = np.load('dataset/TCM_Labels_5.npy')
# print(data1)
# # 输出维度（shape）
# print(f"数组维度: {data.shape}")
# print(f"维度数: {data.ndim}")
# print(f"数据类型: {data.dtype}")

# print(f"数组维度: {data1.shape}")
# print(f"维度数: {data1.ndim}")
# print(f"数据类型: {data1.dtype}")
############################################################################################################################
# import pandas as pd
# import numpy as np
# import json
# from collections import defaultdict

# # ==================== 1. 读取所有数据文件 ====================

# # 读取草药-成分对应关系
# herb_ingredient = pd.read_csv('dataset/hash_herb_ingredient.csv', skiprows=1, header=None, names=['herb', 'ingredient'])

# # 读取草药-靶点对应关系
# herb_target = pd.read_csv('dataset/hash_herb_target.csv', skiprows=1, header=None, names=['herb', 'target'])

# # 读取草药特征矩阵 (252, 300)
# herb_features = np.load('dataset/herb_features_.npy')

# # 读取毒性标签矩阵 (252, 5)
# tcm_labels = np.load('dataset/TCM_Labels_5.npy')

# # 读取草药编号映射
# with open('dataset/hash_herb.json', 'r', encoding='utf-8') as f:
#     herb_to_idx = json.load(f)

# print(f"草药-成分对数: {len(herb_ingredient)}")
# print(f"草药-靶点对数: {len(herb_target)}")
# print(f"特征矩阵形状: {herb_features.shape}")
# print(f"标签矩阵形状: {tcm_labels.shape}")
# print(f"草药数量: {len(herb_to_idx)}")

# # ==================== 2. 数据整合 ====================

# # 创建结果字典
# result_data = []

# # 遍历所有草药（按编号顺序）
# for herb_name, idx in sorted(herb_to_idx.items(), key=lambda x: x[1]):
    
#     # 获取该草药的所有成分（去重）
#     ingredients = set(herb_ingredient[herb_ingredient['herb'] == herb_name]['ingredient'].tolist())
    
#     # 获取该草药的所有靶点（去重）
#     targets = set(herb_target[herb_target['herb'] == herb_name]['target'].tolist())
    
#     # 获取特征向量（300维01数组），转为索引集合
#     feature_vector = herb_features[idx]
#     feature_indices = set(np.where(feature_vector == 1)[0].tolist())
    
#     # 获取毒性标签（5维01数组）
#     toxicity_label = tcm_labels[idx].tolist()
    
#     # 添加到结果
#     result_data.append({
#         '草药名称': herb_name,
#         '成分集合': ingredients,
#         '靶点集合': targets,
#         '特征集合': feature_indices,
#         '毒性标签': toxicity_label
#     })

# # ==================== 3. 创建DataFrame并保存 ====================

# df = pd.DataFrame(result_data)

# # 将集合转为字符串以便CSV存储（可选）
# df['成分集合'] = df['成分集合'].apply(lambda x: ';'.join(sorted(x)) if x else '')
# df['靶点集合'] = df['靶点集合'].apply(lambda x: ';'.join(sorted(x)) if x else '')
# df['特征集合'] = df['特征集合'].apply(lambda x: ';'.join(map(str, sorted(x))) if x else '')
# df['毒性标签'] = df['毒性标签'].apply(lambda x: ','.join(map(str, x)))

# # 保存为CSV
# df.to_csv('dataset/out/integrated_herb_data.csv', index=False, encoding='utf-8-sig')

# print(f"\n整合完成！共 {len(df)} 种草药")
# print(f"CSV文件已保存: dataset/out/integrated_herb_data.csv")
# print("\n数据预览：")
# print(df.head())
############################################################################################################################
import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

# ==================== 配置路径 ====================
INPUT_DIR = 'dataset/'  # 根据你的实际路径修改
OUTPUT_DIR = 'dataset/output'

# 文件路径
HERB_INGREDIENT_CSV = os.path.join(INPUT_DIR, 'hash_herb_ingredient.csv')
HERB_TARGET_CSV = os.path.join(INPUT_DIR, 'hash_herb_target.csv')
INGREDIENT_TARGET_CSV = os.path.join(INPUT_DIR, 'hash_ingredient_target.csv')
HERB_FEATURES_NPY = os.path.join(INPUT_DIR, 'herb_features_.npy')
TCM_LABELS_NPY = os.path.join(INPUT_DIR, 'TCM_Labels_5.npy')
HERB_JSON = os.path.join(INPUT_DIR, 'hash_herb.json')

# ==================== 1. 读取所有数据文件 ====================
print("="*70)
print("步骤1: 读取数据文件")
print("="*70)

# 读取草药-成分对应关系
herb_ingredient_df = pd.read_csv(HERB_INGREDIENT_CSV, skiprows=1, header=None, names=['herb', 'ingredient'])
print(f"✓ 草药-成分对数: {len(herb_ingredient_df)}")

# 读取草药-靶点对应关系  
herb_target_df = pd.read_csv(HERB_TARGET_CSV, skiprows=1, header=None, names=['herb', 'target'])
print(f"✓ 草药-靶点对数: {len(herb_target_df)}")

# 读取成分-靶点作用关系（全局关系）
ingredient_target_df = pd.read_csv(INGREDIENT_TARGET_CSV, skiprows=1, header=None, names=['ingredient', 'target'])
print(f"✓ 成分-靶点作用对数: {len(ingredient_target_df)}")

# 读取草药特征矩阵
herb_features = np.load(HERB_FEATURES_NPY)
print(f"✓ 特征矩阵形状: {herb_features.shape}")

# 读取毒性标签矩阵
tcm_labels = np.load(TCM_LABELS_NPY)
print(f"✓ 标签矩阵形状: {tcm_labels.shape}")

# 读取草药编号映射
with open(HERB_JSON, 'r', encoding='utf-8') as f:
    herb_to_idx = json.load(f)
print(f"✓ 草药数量: {len(herb_to_idx)}")

# ==================== 2. 构建全局编号映射 ====================
print("\n" + "="*70)
print("步骤2: 构建成分和靶点的全局编号")
print("="*70)

# 获取所有唯一成分（从ingredient_target_df中，因为这是全局关系的基准）
all_compounds = sorted(ingredient_target_df['ingredient'].unique())
compound2id = {compound: idx for idx, compound in enumerate(all_compounds)}
print(f"✓ 唯一成分数: {len(all_compounds)}")

# 获取所有唯一靶点（从ingredient_target_df中）
all_targets = sorted(ingredient_target_df['target'].unique())
target2id = {target: idx for idx, target in enumerate(all_targets)}
print(f"✓ 唯一靶点数: {len(all_targets)}")

# 保存编号映射
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'compound2id.json'), 'w', encoding='utf-8') as f:
    json.dump(compound2id, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTPUT_DIR, 'target2id.json'), 'w', encoding='utf-8') as f:
    json.dump(target2id, f, ensure_ascii=False, indent=2)
print(f"✓ 编号映射已保存到 {OUTPUT_DIR}")

# ==================== 3. 构建全局成分-靶点关系集合 ====================
print("\n" + "="*70)
print("步骤3: 构建全局成分-靶点关系集合")
print("="*70)

# 构建全局关系集合 (compound_id, target_id)
global_relations = set()
for _, row in ingredient_target_df.iterrows():
    compound_id = compound2id[row['ingredient']]
    target_id = target2id[row['target']]
    global_relations.add((compound_id, target_id))

print(f"✓ 全局关系数: {len(global_relations)}")

# ==================== 4. 处理每个草药的数据 ====================
print("\n" + "="*70)
print("步骤4: 处理每个草药的数据")
print("="*70)

# 按草药聚合数据
herb_to_compounds = defaultdict(set)
for _, row in herb_ingredient_df.iterrows():
    herb_to_compounds[row['herb']].add(row['ingredient'])

herb_to_targets = defaultdict(set)
for _, row in herb_target_df.iterrows():
    herb_to_targets[row['herb']].add(row['target'])

# 构建每个草药的JSON数据
herb_data_list = []

for herb_name, idx in sorted(herb_to_idx.items(), key=lambda x: x[1]):
    # 获取该草药的成分集合（转为全局ID）
    compounds = herb_to_compounds.get(herb_name, set())
    compound_ids = sorted([compound2id[c] for c in compounds if c in compound2id])
    
    # 获取该草药的靶点集合（转为全局ID）
    targets = herb_to_targets.get(herb_name, set())
    target_ids = sorted([target2id[t] for t in targets if t in target2id])
    
    # 如果没有成分或靶点，跳过
    if len(compound_ids) == 0 or len(target_ids) == 0:
        print(f"⚠ 警告: {herb_name} 缺少成分或靶点，跳过")
        continue
    
    # 构建交互矩阵
    interaction_matrix = np.zeros((len(compound_ids), len(target_ids)), dtype=int)
    
    # 填充矩阵
    for i, comp_id in enumerate(compound_ids):
        for j, targ_id in enumerate(target_ids):
            if (comp_id, targ_id) in global_relations:
                interaction_matrix[i][j] = 1
    
    # 获取特征（300维中值为1的索引）
    feature_vector = herb_features[idx]
    feature_ids = np.where(feature_vector == 1)[0].tolist()
    
    # 获取标签
    label = tcm_labels[idx].tolist()
    
    # 构建数据字典
    herb_data = {
        "herb_name": herb_name,
        "compound_ids": compound_ids,
        "target_ids": target_ids,
        "interaction_matrix": interaction_matrix.tolist(),
        "feature_ids": feature_ids,
        "feature_vector": feature_vector.tolist(),
        "label": label
    }
    herb_data_list.append(herb_data)

print(f"✓ 成功处理草药数: {len(herb_data_list)}")

# ==================== 5. 划分训练集/验证集/测试集 ====================
print("\n" + "="*70)
print("步骤5: 划分训练集/验证集/测试集 (7:1.5:1.5)")
print("="*70)

# 多标签 stratify：将标签组合转为字符串
label_strs = [''.join(str(int(v)) for v in d['label']) for d in herb_data_list]
label_counts = Counter(label_strs)
can_stratify = min(label_counts.values()) >= 2

# 第一步：分出 train (70%) vs temp (30%)
if can_stratify:
    train_data, temp_data, _, temp_strs = train_test_split(
        herb_data_list, label_strs,
        test_size=0.3, random_state=42, stratify=label_strs
    )
else:
    train_data, temp_data = train_test_split(
        herb_data_list, test_size=0.3, random_state=42
    )
    temp_strs = [''.join(str(int(v)) for v in d['label']) for d in temp_data]

temp_counts = Counter(temp_strs)
can_stratify_temp = min(temp_counts.values()) >= 2

# 第二步：temp 对半分为 val (15%) 和 test (15%)
if can_stratify_temp:
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42, stratify=temp_strs
    )
else:
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42
    )

print(f"✓ 训练集样本数: {len(train_data)} ({len(train_data)/len(herb_data_list)*100:.1f}%)")
print(f"✓ 验证集样本数: {len(val_data)} ({len(val_data)/len(herb_data_list)*100:.1f}%)")
print(f"✓ 测试集样本数: {len(test_data)} ({len(test_data)/len(herb_data_list)*100:.1f}%)")

# ==================== 6. 保存为JSON文件 ====================
print("\n" + "="*70)
print("步骤6: 保存JSON文件")
print("="*70)

# 创建目录
train_dir = os.path.join(OUTPUT_DIR, 'train')
val_dir = os.path.join(OUTPUT_DIR, 'val')
test_dir = os.path.join(OUTPUT_DIR, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 保存训练集
train_json_path = os.path.join(train_dir, 'train.json')
with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
print(f"✓ 训练集已保存: {train_json_path}")

# 保存验证集
val_json_path = os.path.join(val_dir, 'val.json')
with open(val_json_path, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)
print(f"✓ 验证集已保存: {val_json_path}")

# 保存测试集
test_json_path = os.path.join(test_dir, 'test.json')
with open(test_json_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
print(f"✓ 测试集已保存: {test_json_path}")

# 保存全量数据（供 K 折交叉验证使用）
all_json_path = os.path.join(OUTPUT_DIR, 'all_herbs.json')
with open(all_json_path, 'w', encoding='utf-8') as f:
    json.dump(herb_data_list, f, ensure_ascii=False, indent=2)
print(f"✓ 全量数据已保存: {all_json_path}")

# ==================== 7. 保存划分信息和统计 ====================
print("\n" + "="*70)
print("步骤7: 保存统计信息")
print("="*70)

stats = {
    "total_herbs": len(herb_data_list),
    "train_herbs": len(train_data),
    "val_herbs": len(val_data),
    "test_herbs": len(test_data),
    "total_compounds": len(compound2id),
    "total_targets": len(target2id),
    "feature_dims": int(herb_features.shape[1]),
    "global_relations": len(global_relations),
    "train_herb_names": [d["herb_name"] for d in train_data],
    "val_herb_names": [d["herb_name"] for d in val_data],
    "test_herb_names": [d["herb_name"] for d in test_data],
    "random_state": 42,
    "split_ratio": "7:1.5:1.5"
}

stats_path = os.path.join(OUTPUT_DIR, 'statistics.json')
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"✓ 统计信息已保存: {stats_path}")

# ==================== 8. 显示样本示例 ====================
print("\n" + "="*70)
print("步骤8: 数据示例")
print("="*70)

print("\n训练集第一个样本:")
sample = train_data[0]
print(f"  草药名称: {sample['herb_name']}")
print(f"  成分数: {len(sample['compound_ids'])}")
print(f"  靶点数: {len(sample['target_ids'])}")
print(f"  交互矩阵形状: ({len(sample['interaction_matrix'])}, {len(sample['interaction_matrix'][0])})")
print(f"  特征数: {len(sample['feature_ids'])}")
print(f"  标签: {sample['label']}")

print("\n验证集第一个样本:")
sample = val_data[0]
print(f"  草药名称: {sample['herb_name']}")
print(f"  成分数: {len(sample['compound_ids'])}")
print(f"  靶点数: {len(sample['target_ids'])}")
print(f"  交互矩阵形状: ({len(sample['interaction_matrix'])}, {len(sample['interaction_matrix'][0])})")
print(f"  特征数: {len(sample['feature_ids'])}")
print(f"  标签: {sample['label']}")

print("\n" + "="*70)
print("处理完成！")
print("="*70)
print(f"\n输出文件:")
print(f"  {OUTPUT_DIR}/")
print(f"  ├── compound2id.json      (成分编号映射)")
print(f"  ├── target2id.json        (靶点编号映射)")
print(f"  ├── statistics.json       (统计信息)")
print(f"  ├── train/")
print(f"  │   └── train.json        (训练集，{len(train_data)}个样本)")
print(f"  ├── val/")
print(f"  │   └── val.json          (验证集，{len(val_data)}个样本)")
print(f"  └── test/")
print(f"      └── test.json         (测试集，{len(test_data)}个样本)")

# ==================== 9. 输出格式说明 ====================
print("\n" + "="*70)
print("输出JSON格式说明:")
print("="*70)
print("""
每个样本包含以下字段:
- herb_name: 草药名称 (字符串)
- compound_ids: 成分ID列表 (整数列表)，全局编号
- target_ids: 靶点ID列表 (整数列表)，全局编号  
- interaction_matrix: 二维列表 (compound_ids × target_ids)，0/1表示是否有作用关系
- feature_ids: 特征ID列表 (整数列表)，herb_features_中值为1的索引
- label: 毒性标签列表 (5个0/1值)，对应5种毒性类型
""")