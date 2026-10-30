"""训练 BPE 词表并把文本编码成 uint16 token 数组（.npy，训练时用 memmap 读取）。

例子：
    uv run python experiments/prepare_data.py \
        --train-text data/TinyStoriesV2-GPT4-train.txt --valid-text data/TinyStoriesV2-GPT4-valid.txt \
        --out-dir data/tinystories --vocab-size 10000 --bpe-sample-mb 50 --train-mb 200

只编码训练集的前 --train-mb MB：讲义低资源配置只需要约 41M token，
TinyStories 在 10k 词表下大约每个 token 4 字节，200MB 足够，不必把 2GB 全部编码。
"""

import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cs336_basics.bpe import Tokenizer, train_bpe

SPECIAL_TOKEN = "<|endoftext|>"

_tokenizer = None


def read_prefix(path, max_bytes):
    # 读前 max_bytes 字节，并截到最后一个完整文档，避免把一篇故事切成两半。
    with open(path, "rb") as f:
        data = f.read(max_bytes) if max_bytes else f.read()
    if max_bytes and len(data) == max_bytes:
        cut = data.rfind(SPECIAL_TOKEN.encode())
        if cut > 0:
            data = data[: cut + len(SPECIAL_TOKEN)]
    return data.decode("utf-8", errors="ignore")


def split_into_chunks(text, num_chunks):
    # 按文档边界切成 num_chunks 份，交给多个进程编码；在 special token 处切，结果与整体编码一致。
    documents = text.split(SPECIAL_TOKEN)
    per_chunk = max(1, len(documents) // num_chunks)
    chunks = []
    for i in range(0, len(documents), per_chunk):
        chunks.append(SPECIAL_TOKEN.join(documents[i : i + per_chunk]))
    # 被切开的位置补回 special token
    return [chunk + SPECIAL_TOKEN if i < len(chunks) - 1 else chunk for i, chunk in enumerate(chunks)]


def _init_worker(tokenizer):
    global _tokenizer
    _tokenizer = tokenizer


def _encode(chunk):
    return np.asarray(_tokenizer.encode(chunk), dtype=np.uint16)


def encode_text(tokenizer, text, workers):
    chunks = split_into_chunks(text, workers * 8)
    with Pool(workers, initializer=_init_worker, initargs=(tokenizer,)) as pool:
        parts = pool.map(_encode, chunks)
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--valid-text", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--bpe-sample-mb", type=float, default=50, help="训练 BPE 用的语料大小")
    parser.add_argument("--train-mb", type=float, default=200, help="编码多少 MB 训练文本；0 表示全部")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    assert args.vocab_size <= 2**16, "uint16 只能放 65536 个 token id"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args)}

    start = time.perf_counter()
    sample_path = out_dir / "bpe_sample.txt"
    sample_path.write_text(read_prefix(args.train_text, int(args.bpe_sample_mb * 2**20)), encoding="utf-8")
    vocab, merges = train_bpe(sample_path, args.vocab_size, [SPECIAL_TOKEN])
    sample_path.unlink()
    with open(out_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": [SPECIAL_TOKEN]}, f)
    meta["bpe_seconds"] = time.perf_counter() - start
    print(f"BPE: {len(vocab)} tokens in {meta['bpe_seconds']:.0f}s")

    tokenizer = Tokenizer(vocab, merges, [SPECIAL_TOKEN])
    for split, path, max_mb in (("train", args.train_text, args.train_mb), ("valid", args.valid_text, 0)):
        start = time.perf_counter()
        text = read_prefix(path, int(max_mb * 2**20))
        ids = encode_text(tokenizer, text, args.workers)
        np.save(out_dir / f"{split}.npy", ids)
        meta[split] = {
            "bytes": len(text.encode("utf-8")),
            "tokens": int(len(ids)),
            "bytes_per_token": len(text.encode("utf-8")) / len(ids),
            "seconds": time.perf_counter() - start,
        }
        print(split, meta[split])

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
