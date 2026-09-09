import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import os

def get_scaffold(smiles):
    """분자의 SMILES로부터 Bemis-Murcko Scaffold를 추출"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # 원자 기호와 결합 정보를 유지한 스캐폴드 추출
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=True)
    except:
        return None
    return None

# 1. 파일 경로 설정 (이전에 필터링된 파일 사용)
data_path = 'data/pubchem/data_filtered.csv'
output_path = os.path.join(os.path.dirname(data_path), 'data_scaffold_5fold.csv')

# 2. 데이터 로드
df = pd.read_csv(data_path)
print(f"전체 데이터 개수: {len(df)}")

# 3. 스캐폴드별로 인덱스 그룹화
scaffold_to_indices = defaultdict(list)
for idx, smiles in enumerate(df['smiles']):
    scaf = get_scaffold(smiles)
    if scaf is None:
        scaf = 'invalid' # 유효하지 않은 SMILES 처리
    scaffold_to_indices[scaf].append(idx)

# 4. 균등한 분배를 위해 스캐폴드 뭉치 크기순으로 정렬
sorted_scaffolds = sorted(scaffold_to_indices.values(), key=len, reverse=True)

# 5. Greedy Bin-packing 알고리즘으로 분배 로직 수정
K = 5
fold_indices = [[] for _ in range(K)]
fold_counts = [0] * K  # 각 Fold의 현재 데이터 개수 추적

for idx_list in sorted_scaffolds:
    # 현재 가장 데이터가 적은 Fold의 인덱스를 찾음
    min_fold_idx = fold_counts.index(min(fold_counts))

    # 해당 Fold에 추가
    fold_indices[min_fold_idx].extend(idx_list)
    fold_counts[min_fold_idx] += len(idx_list)

# 6. 결과 기록
df['fold'] = -1
for f_idx, indices in enumerate(fold_indices):
    df.loc[indices, 'fold'] = f_idx

# 7. 결과 저장 및 확인
df.to_csv(output_path, index=False)

print("\n" + "="*40)
print(f"Scaffold K-fold 분할 결과 (K={K})")
for i in range(K):
    print(f"Fold {i}: {len(df[df['fold']==i])} 개")
print(f"저장 완료: {output_path}")
print("="*40)
