from transformers import StoppingCriteria, StoppingCriteriaList

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_ids):
        self.tokenizer = tokenizer
        self.stop_ids = stop_ids  # 리스트로 여러 개 가능

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids: (batch, sequence)
        # 마지막 토큰이 stop_ids 중 하나면 True 반환 → 멈춤
        for stop_id in self.stop_ids:
            if input_ids[0, -1].item() == stop_id:
                return True
        return False

# 사용 예시
newline_token_id = tokenizer.convert_tokens_to_ids('\n')
custom_eos_id = tokenizer.eos_token_id  # 기존 EOS도 같이
stop_ids = [newline_token_id, custom_eos_id]

stopping_criteria = StoppingCriteriaList([MyStoppingCriteria(tokenizer, stop_ids)])
output = model.generate(
    input_ids,
    stopping_criteria=stopping_criteria,
    # max_new_tokens, do_sample 등 필요에 따라 추가
)
