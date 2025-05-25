# saves the wikipedia dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # loads wikipedia dataset from huggingface
    # this is the 20220301.simple dataset which is smaller and in simple English
    dataset = load_dataset("wikipedia", "20220301.simple", num_proc=num_proc_load_dataset)

    # wikipedia by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['id', 'url', 'title', 'text'],
    #         num_rows: ~200k (varies by dataset)
    #     })
    #     val: Dataset({
    #         features: ['id', 'url', 'title', 'text'],
    #         num_rows: ~100 (varies by dataset)
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        # combine title and text with a separator
        full_text = example['title'] + '\n\n' + example['text']
        ids = enc.encode_ordinary(full_text) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['id', 'url', 'title', 'text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"Total number of tokens in {split} split: {arr_len}")
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16                    # enc.max_token_value < 2**16
        arr   = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        # never create more contiguous shards than there are rows
        total_batches = min(1024, len(dset))    # <-- key line

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = (
                dset
                .shard(num_shards=total_batches, index=batch_idx, contiguous=True)
                .with_format("numpy")
            )
            if len(batch) == 0:                 # guard against any empty shard
                continue
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
    
    # train.bin and val.bin sizes will vary depending on the wikipedia dataset chosen
    # Simple English Wikipedia is much smaller than full Wikipedia

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')