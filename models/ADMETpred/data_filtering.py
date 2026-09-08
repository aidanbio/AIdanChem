import pandas as pd
from tdc.benchmark_group import admet_group
from rdkit import Chem
import os


def canonicalize_smiles(smiles):
    """SMILES를 표준화(isomericSmiles 포함)하여 중복 체크 정확도 향상"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # 기본적으로 RDKit의 Canonicalization 알고리즘이 적용됩니다.
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None
    return None


# 1. 파일 경로 설정
# 22개 속성 데이터셋 혹은 41개 속성 데이터셋 중 하나를 선택하세요.
data_path = 'data/pubchem/data.csv'

output_filename = os.path.join(os.path.dirname(data_path), 'data_filtered.csv')

root = './'
tdc_name_path = os.path.join(root, 'tdc_name.txt')

# 2. tdc_name.txt 로드
# eval_finetuned_tdc_2.py에서 사용한 것과 동일한 매핑 파일을 사용합니다.
tdc_info = pd.read_csv(tdc_name_path)
print(f"로드된 평가 특성 개수: {len(tdc_info)}")

# 3. 모든 TDC 벤치마크 테스트 데이터의 SMILES 수집
test_smiles_set = set()
group = admet_group(path='data/')

print("TDC 벤치마크 테스트 데이터 수집 및 표준화 중...")
for _, row in tdc_info.iterrows():
    feature_name = row['feature']
    tdc_dataset_name = row['tdc_name']

    try:
        benchmark = group.get(tdc_dataset_name)
        # 벤치마크 그룹의 'test' 세트에서 'Drug' 컬럼(SMILES)을 가져옵니다.
        test_list = benchmark['test']['Drug'].tolist()

        for s in test_list:
            can_s = canonicalize_smiles(s)
            if can_s:
                test_smiles_set.add(can_s)
    except Exception as e:
        print(f"⚠️ {feature_name} ({tdc_dataset_name}) 로드 실패: {e}")

print(f"총 수집된 고유 벤치마크 테스트 분자 수: {len(test_smiles_set)}")

# 4. 내 데이터 필터링
print(f"\n내 데이터 로드 중: {data_path}")
df_all = pd.read_csv(data_path)
initial_count = len(df_all)

# 내 데이터의 SMILES 표준화 (필터링용 임시 컬럼)
print("내 데이터 SMILES 표준화 진행 중 (시간이 다소 소요될 수 있습니다)...")
df_all['canonical_smiles'] = df_all['smiles'].apply(canonicalize_smiles)

# 벤치마크 세트에 포함된 분자들을 제외
df_filtered = df_all[~df_all['canonical_smiles'].isin(test_smiles_set)].copy()

# 5. 결과 저장
# df_filtered = df_filtered.drop(columns=['canonical_smiles'])
# df_filtered.to_csv(output_filename, index=False)
df_filtered['smiles'] = df_filtered['canonical_smiles']
df_filtered = df_filtered.drop(columns=['canonical_smiles'])
df_filtered.to_csv(output_filename, index=False)

print("\n" + "="*40)
print(f"필터링 결과 보고")
print(f"원본 데이터: {initial_count} 개")
print(f"최종 데이터: {len(df_filtered)} 개")
print(f"제거된 데이터: {initial_count - len(df_filtered)} 개")
print(f"저장 경로: {os.getcwd()}/{output_filename}")
print("="*40)
