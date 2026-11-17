"""交替对比：比较几个配置的速度时，怎么安排运行顺序、怎么汇总。

为什么不能“先把 A 跑完再跑 B”：笔记本 GPU 的时钟会随温度和功耗墙漂移。
A 全在机器凉的时候跑、B 全在机器热的时候跑，测出的差异里就混进了环境的差异。
交替跑（A B A B ...）让每个配置都经历相近的环境，再对多轮结果取平均和标准差。
做法参考 modded-nanogpt 纪录 #92 的验证方式（新旧版本在同一台机器、同一时段交替运行）
和 MLPerf Training 规则（多次运行、不许挑最好的一次）。
"""

import math


def interleaved_order(names, rounds):
    """返回 [(round, name), ...]。每一轮每个配置各跑一次，并且每轮轮换起始位置。

    轮换是为了不让某个配置总是排在第一个：每轮的第一个往往最凉、最快。
    例如 names=[A, B], rounds=3 -> A B | B A | A B
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if len(set(names)) != len(names):
        raise ValueError("config names must be unique")
    order = []
    for round_index in range(rounds):
        shift = round_index % len(names)
        for name in names[shift:] + names[:shift]:
            order.append((round_index, name))
    return order


def mean_std(values):
    """均值和样本标准差（除以 n-1）。只有一个值时标准差没有意义，返回 None。"""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def summarize(records, metric="step_ms", baseline=None):
    """按配置汇总多轮结果。

    records: 每个元素至少有 "config"、"round" 和 metric 这几个键（OOM 的记录 metric 为 None，会被跳过）。
    baseline: 作为比较对象的配置名，默认是第一个出现的配置。
    返回的每一行里：
      mean / std   这个配置 metric 的均值和标准差
      ratio_mean / ratio_std
                   同一轮里 baseline 的值除以本配置的值（metric 是耗时的话，大于 1 表示比 baseline 快），
                   再对各轮取均值和标准差。按轮配对是为了抵消轮与轮之间的环境漂移。
    """
    by_config = {}
    for record in records:
        if record.get(metric) is None:
            continue
        by_config.setdefault(record["config"], {})[record["round"]] = record[metric]
    if not by_config:
        return []
    if baseline is None:
        baseline = next(iter(by_config))
    if baseline not in by_config:
        raise ValueError(f"baseline {baseline!r} has no valid records")

    rows = []
    for name, by_round in by_config.items():
        mean, std = mean_std(list(by_round.values()))
        shared_rounds = sorted(set(by_round) & set(by_config[baseline]))
        ratios = [by_config[baseline][r] / by_round[r] for r in shared_rounds]
        ratio_mean, ratio_std = mean_std(ratios) if ratios else (None, None)
        rows.append(
            {
                "config": name,
                "n": len(by_round),
                "mean": mean,
                "std": std,
                "min": min(by_round.values()),
                "max": max(by_round.values()),
                "ratio_mean": ratio_mean,
                "ratio_std": ratio_std,
            }
        )
    return rows


def format_markdown(rows, metric="step_ms"):
    def fmt(value, digits):
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [
        f"| 配置 | 轮数 | {metric} 均值 | 标准差 | 最小 | 最大 | 相对基线（均值 ± 标准差） |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['n']} | {fmt(row['mean'], 2)} | {fmt(row['std'], 2)} "
            f"| {fmt(row['min'], 2)} | {fmt(row['max'], 2)} "
            f"| {fmt(row['ratio_mean'], 3)} ± {fmt(row['ratio_std'], 3)} |"
        )
    return "\n".join(lines)
