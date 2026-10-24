from collections import Counter, defaultdict
import regex as re


# GPT-2 风格 pre-tokenization 规则：
# 在真正做 BPE 之前，先把原始文本粗切成 pieces。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path, vocab_size, special_tokens):
    # vocab: token id -> token bytes
    # 初始放入 256 个单 byte token，保证任意 UTF-8 文本都能被表示。
    vocab = {}
    
    for i in range(256):
        vocab[i] = bytes([i])

    # special tokens 直接加入 vocab，不参与普通 BPE merge。
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    
    # 读取训练语料。
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    pieces = []

    # 先按 special tokens 切开文本。
    # 普通 chunk 做 pre-tokenization；special token chunk 跳过，
    # 防止 BPE 学到 b"<|" 这类 special token 的碎片。
    if special_tokens:
        special_pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        chunks = re.split(special_pattern, text)

        for chunk in chunks:
            if chunk == "":
                continue

            if chunk in special_tokens:
                continue

            pieces.extend(re.findall(PAT, chunk))
    else:
        pieces = re.findall(PAT, text)
    
    # piece_counts:
    # 统计每种 pre-token piece 出现多少次。
    # 例如 "the" -> (b"t", b"h", b"e")。    
    piece_counts = Counter()
    
    for piece in pieces:
        piece_bytes = piece.encode("utf-8")
        piece_tuple = tuple(bytes([b]) for b in piece_bytes)
        piece_counts[piece_tuple] += 1

    # 给每个 piece 一个 id，方便建立 pair -> pieces 的反向索引。
    #
    # piece_tokens: piece_id -> 当前这个 piece 的 token 序列
    # piece_freqs:  piece_id -> 这个 piece 在语料中出现多少次
    piece_tokens = {}
    piece_freqs = {}    

    for piece_id, (piece_tuple, count) in enumerate(piece_counts.items()):
        piece_tokens[piece_id] = piece_tuple
        piece_freqs[piece_id] = count
    
    # merges 记录 BPE 学到的合并规则，顺序很重要。
    merges = []
    
    # pair_counts:
    # 当前所有相邻 pair 的总频率。
    #
    # pair_to_pieces:
    # 每个 pair 出现在哪些 piece 里。
    # 这是加速的核心：merge 时只更新包含 best_pair 的 pieces。
    pair_counts = Counter()
    pair_to_pieces = defaultdict(set)
    
    # 初始化 pair_counts 和 pair_to_pieces。
    # 如果一个 piece 出现 count 次，那么它内部每个相邻 pair 都贡献 count 次。
    for piece_id, piece_tuple in piece_tokens.items():
        count = piece_freqs[piece_id]

        for i in range(len(piece_tuple) - 1):
            pair = (piece_tuple[i], piece_tuple[i + 1])
            pair_counts[pair] += count
            pair_to_pieces[pair].add(piece_id)
    
    # 每轮 merge 都要维护三个状态一致：
    # 1. piece_tokens：当前每个 piece 长什么样
    # 2. pair_counts：当前每个 pair 出现多少次
    # 3. pair_to_pieces：当前每个 pair 出现在哪些 pieces 里
    #
    # 更新顺序必须是：
    # 删旧统计 -> 改 piece -> 加新统计
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        
        # 选择当前最高频 pair。
        # 如果频率相同，用 pair 本身做 tie-break，保证结果稳定。    
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        
        # 记录这次 merge，并把新 token 加入 vocab。
        merges.append(best_pair)
        new_token = b''.join(best_pair)
        vocab[len(vocab)] = new_token
        
        # 只处理包含 best_pair 的 pieces。
        # 不包含 best_pair 的 piece 不会变化，所以不用扫描。
        affected_piece_ids = list(pair_to_pieces[best_pair])

        for piece_id in affected_piece_ids:
            old_piece_tuple = piece_tokens[piece_id]
            count = piece_freqs[piece_id]
            
            # 这个 piece 马上要被改掉。
            # 所以先撤销它旧的 pair 统计。
            for i in range(len(old_piece_tuple) - 1):
                old_pair = (old_piece_tuple[i], old_piece_tuple[i + 1])
                pair_counts[old_pair] -= count

                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]

                pair_to_pieces[old_pair].discard(piece_id)

            # 在这个 piece 内部真正执行 best_pair 的合并。
            # 命中 best_pair 时，两个 token 合成一个 new_token，所以 i += 2。
            # 没命中时，保留当前 token，所以 i += 1。
            new_piece_tuple = []
            i = 0

            while i < len(old_piece_tuple):
                if i < len(old_piece_tuple) - 1 and (old_piece_tuple[i], old_piece_tuple[i + 1]) == best_pair:
                    new_piece_tuple.append(new_token)
                    i += 2
                else:
                    new_piece_tuple.append(old_piece_tuple[i])
                    i += 1
                    
            # 更新这个 piece 当前的 token 序列。
            # 下一轮 BPE 会基于这个新状态继续训练。
            new_piece_tuple = tuple(new_piece_tuple)
            piece_tokens[piece_id] = new_piece_tuple
            for i in range(len(new_piece_tuple) - 1):
                new_pair = (new_piece_tuple[i], new_piece_tuple[i + 1])
                pair_counts[new_pair] += count
                pair_to_pieces[new_pair].add(piece_id)

    return vocab, merges 


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token_to_id = {}

        for token_id, token_bytes in vocab.items():
            self.token_to_id[token_bytes] = token_id
        
        self.merge_ranks = {}

        for rank, pair in enumerate(merges):
            self.merge_ranks[pair] = rank
    

    def decode(self, ids):

        token_bytes = []

        for token_id in ids:
            token_bytes.append(self.vocab[token_id])

        all_bytes = b"".join(token_bytes)
        return all_bytes.decode("utf-8", errors="replace")
    
    def encode(self, text):
        ids = []
    

        if self.special_tokens:
            # 长的 special token 必须排前面，处理重叠 special token。
            sorted_special_tokens = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )
            special_pattern = (
                "("
                + "|".join(re.escape(token) for token in sorted_special_tokens)
                + ")"
            )
            chunks = re.split(special_pattern, text)
        else:
            chunks = [text]

        for chunk in chunks:
            if chunk == "":
                continue

            # special token 不做 PAT 切分和 BPE merge，直接查它的 id。
            if chunk in self.special_tokens:
                special_bytes = chunk.encode("utf-8")
                ids.append(self.token_to_id[special_bytes])
                continue

            pieces = re.findall(PAT, chunk)

            for piece in pieces:
                piece_bytes = piece.encode("utf-8")
                tokens = [bytes([byte]) for byte in piece_bytes]
            
                while True:
                    best_pair = None
                    best_rank = None

                # 找当前 tokens 中，优先级最高的可合并 pair。
                    for i in range(len(tokens) - 1):
                        pair = (tokens[i], tokens[i + 1])

                        if pair in self.merge_ranks:
                            rank = self.merge_ranks[pair]

                            if best_rank is None or rank < best_rank:
                                best_pair = pair
                                best_rank = rank

                    # 没有任何 pair 能再合并，停止。
                    if best_pair is None:
                        break

                    # 从左到右，把 best_pair 的每个非重叠出现位置合并。
                    new_tokens = []
                    i = 0

                    while i < len(tokens):
                        if (
                            i < len(tokens) - 1
                            and (tokens[i], tokens[i + 1]) == best_pair
                        ):
                            new_tokens.append(b"".join(best_pair))
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1

                    tokens = new_tokens

                # 全部 merge 完后，最终 bytes token 才转换成 ids。
                for token in tokens:
                    ids.append(self.token_to_id[token])
    

        return ids
    
    def encode_iterable(self, iterable):
        for text_chunk in iterable:
            chunk_ids = self.encode(text_chunk)

            for token_id in chunk_ids:
                yield token_id