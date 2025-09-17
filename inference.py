import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import PartialState
from accelerate.utils import gather_object

import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import pickle
import os
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_path", type=str, default='.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name
    dataset = args.dataset

    # Start up the distributed environment without needing the Accelerator.
    distributed_state = PartialState()

    # You can change the model to any LLM such as mistralai/Mistral-7B-v0.1 or meta-llama/Llama-2-7b-chat-hf
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map=distributed_state.device, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,trust_remote_code=True)
    # Need to set the padding token to the eos token for generation
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    try:
        data = load_dataset(args.dataset, split='train')
    except:
        data = load_from_disk(args.dataset)

    # You can change the batch size depending on your GPU RAM
    # We set it to 8 since it is better for some hardware. More information here https://github.com/huggingface/tokenizers/issues/991
    pad_to_multiple_of = 8

    # Split into batches
    # We will get the following results:
    # [ ["I would like to", "hello how are you"], [ "what is going on", "roses are red and"], [ "welcome to the hotel"] ]
    formatted_prompts = [data[i : i + args.batch_size]['text'] for i in range(0, len(data), args.batch_size)]

    # Apply padding on the left since we are doing generation
    padding_side_default = tokenizer.padding_side
    tokenizer.padding_side = "left"
    # Tokenize each batch
    tokenized_prompts = [
        tokenizer(formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
        for formatted_prompt in formatted_prompts
    ]
    # Put back the original padding behavior
    tokenizer.padding_side = padding_side_default

    completions_per_process = []
    # We automatically split the batched data we passed to it across all the processes. We also set apply_padding=True
    # so that the GPUs will have the same number of prompts, and you can then gather the results.
    # For example, if we have 2 gpus, the distribution will be:
    # GPU 0: ["I would like to", "hello how are you"],  "what is going on", "roses are red and"]
    # GPU 1: ["welcome to the hotel"], ["welcome to the hotel"] -> this prompt is duplicated to ensure that all gpus have the same number of prompts
    with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
        for batch in tqdm(batched_prompts):
            # Move the batch to the device
            batch = batch.to(distributed_state.device)
            # batch['labels'] = batch['input_ids']
            # We generate the text, decode it and add it to the list completions_per_process
            #outputs = model.generate(**batch, max_new_tokens=2048)
            bs, seq_len = batch['input_ids'].shape
            with torch.no_grad():
                outputs = model.forward(**batch)
                shift_logits = outputs.logits.view(-1, model.config.vocab_size)
                shift_labels = batch['input_ids'].view(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
                loss = loss.reshape(bs, seq_len) 
                loss = loss.mean(dim=1) # bs
                completions_per_process.extend(loss.cpu().tolist())

    # We are gathering string, so we need to use gather_object.
    # If you need to gather tensors, you can use gather from accelerate.utils
    completions_gather = gather_object(completions_per_process)

    # Drop duplicates produced by apply_padding in split_between_processes
    completions = completions_gather[: len(data)]
    with open(os.path.join(args.output_path, 'loss.pkl'), 'wb') as f:
        pickle.dump(completions, f)
