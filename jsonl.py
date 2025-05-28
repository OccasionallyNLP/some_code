def read_jsonl(file_path):
  import json
  with open(file_path, 'r', encoding='utf-8') as f:
      data = [json.loads(line) for line in f]
  return data

def save_jsonl(file_path, data):
  import json
  """
  data = [
      {"name": "Alice", "age": 30},
      {"name": "Bob", "age": 25}
  ]
  """
  with open(file_path, 'w', encoding='utf-8') as f:
      for item in data:
          f.write(json.dumps(item, ensure_ascii=False) + '\n')
