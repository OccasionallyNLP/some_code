# datasets로 안될 때, jsonl 읽어서 처리하는 방법.
from datasets import Dataset, load_dataset, concatenate_datasets
import os
import argparse
import jsonlines
import parmap
import multiprocessing
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--name", type=str)
parser.add_argument("--output_dir", type=str)

def parse_jsonl(data_file):
    """ JSONL 파일을 로드하여 Dataset 형식으로 변환하는 함수 """
    ds = []
    try:
        with jsonlines.open(data_file) as f:
            for line in f:
                ds.append(dict(id=line['id'], text=line['text']))
        return Dataset.from_list(ds)
    except Exception as e:
        print(f"Error processing {data_file}: {e}")
        return None

if __name__ == "__main__":
    args = parser.parse_args()
    data_paths = [os.path.join(args.data_dir, i) for i in os.listdir(args.data_dir) if i.endswith("jsonl")]

    num_proc = int(os.cpu_count() * 0.95)
    _ds = []

    # load_dataset 시도 후 실패하면 multiprocessing을 사용하여 jsonlines 처리
    failed_files = []
    for data_file in tqdm(data_paths, desc="Processing JSONL Files"):
        try:
            ds = load_dataset("json", data_files=data_file, num_proc=num_proc, columns=['text', 'id'], split='train')
            ds = ds.remove_columns(columns=['metadata'])
            _ds.append(ds)
        except:
            failed_files.append(data_file)  # 실패한 파일들을 저장

    # parmap을 사용한 병렬 처리 (진행률 & 예상 시간 표시)
    print(f"Processing {len(failed_files)} failed JSONL files using parmap...")
    results = parmap.map(parse_jsonl, failed_files, pm_pbar=True, pm_processes=num_proc)

    # None이 아닌 데이터만 필터링하여 데이터셋 생성
    final_datasets = [ds for ds in results if ds is not None]

    # 데이터셋 병합
    final = concatenate_datasets(_ds + final_datasets)
    print("Processing done!")

    # train-test split
    final = final.train_test_split(test_size=0.01)
    print(final)

    # JSONL로 저장
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    final['train'].to_json(os.path.join(args.output_dir, "train", args.name + ".jsonl"), num_proc=num_proc)
