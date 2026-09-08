"""目标分布校准、ESS约束、代表性选择与无泄漏批次。"""

from collections import Counter, defaultdict
import numpy as np


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / max(w @ w, 1e-12))


def gini(w):
    w = np.sort(np.asarray(w, float))
    n = len(w)
    return (
        float((2 * np.arange(1, n + 1) - n - 1) @ w / (n * w.sum()))
        if n and w.sum()
        else 0.0
    )


def target_distribution(application, benchmark, errors, alpha=1.0, beta=0.3, gamma=0.5):
    keys = sorted(set(application) | set(benchmark))
    scores = {
        k: max(application.get(k, 0), 1e-8) ** alpha
        * max(benchmark.get(k, 0), 1e-8) ** beta
        * np.exp(gamma * errors.get(k, 0))
        for k in keys
    }
    total = sum(scores.values())
    return {k: float(v / total) for k, v in scores.items()}


def calibrate_weights(records, targets, min_ess_ratio=0.5, cap=4.0, iterations=100):
    """迭代比例拟合；不可用监督权重为零；通过向均匀分布收缩满足ESS。"""
    valid = np.array(
        [
            r.get("status", "valid") not in ("suspect", "invalid")
            and r.get("quality", 1) > 0
            for r in records
        ]
    )
    if not valid.any():
        raise ValueError("没有有效训练样本")
    w = valid.astype(float)
    for _ in range(iterations):
        for key, target in targets.items():
            for value, mass in target.items():
                mask = (
                    np.array([str(r.get(key)) == str(value) for r in records]) & valid
                )
                if mass > 0 and not mask.any():
                    raise ValueError(f"目标切片无样本：{key}={value}")
                if mask.any():
                    w[mask] *= float(mass) * w.sum() / max(w[mask].sum(), 1e-12)
        w = np.minimum(w, cap)
        w *= valid.sum() / max(w.sum(), 1e-12)
    # 有界且均值为1；收缩会牺牲部分目标拟合，诊断报告显式给出剩余差距。
    for _ in range(200):
        if ess(w) >= min_ess_ratio * valid.sum() and w.max() <= cap + 1e-8:
            break
        w = 0.95 * w + 0.05 * valid
    return w


def largest_remainder(probabilities, size):
    keys = sorted(probabilities)
    p = np.array([probabilities[k] for k in keys], float)
    p /= p.sum()
    real = p * size
    counts = np.floor(real).astype(int)
    for i in np.argsort(-(real - counts), kind="stable")[: size - counts.sum()]:
        counts[i] += 1
    return dict(zip(keys, counts.tolist()))


def representative_indices(features, values, count, seed=3407, diversity=1.0):
    """贪心价值+最远点覆盖；同价值时从最接近中心的代表样本开始。"""
    x = np.asarray(features, float)
    v = np.asarray(values, float)
    if count >= len(x):
        return list(range(len(x)))
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)
    selected = []
    distance = np.full(len(x), np.inf)
    available = np.ones(len(x), bool)
    for step in range(count):
        score = (
            v - np.linalg.norm(x, axis=1) * 0.01
            if step == 0
            else v + diversity * np.minimum(distance, 10.0)
        )
        score[~available] = -np.inf
        i = int(score.argmax())
        selected.append(i)
        available[i] = False
        distance = np.minimum(distance, np.linalg.norm(x - x[i], axis=1))
    return selected


def balanced_batches(records, weights, batch_size=8, seed=3407, repeat_fraction=1.0):
    """每更新RGB/IR等量；域内视频配额均衡；先完整覆盖，再加权重复。"""
    if batch_size % 2:
        raise ValueError("双域宏批次必须为偶数")
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, r in enumerate(records):
        if weights[i] > 0:
            groups[(r["domain"], r["video"])].append(i)
    domains = ("RGB_out", "IR_in")
    if any(not any(d == k[0] for k in groups) for d in domains):
        raise ValueError("双域批次缺少一个域")
    queues = {}
    for key, ids in groups.items():
        base = list(map(int, rng.permutation(ids)))
        p = np.asarray(weights)[ids]
        p = p / p.sum()
        extras = list(
            map(int, rng.choice(ids, int(round(len(ids) * repeat_fraction)), p=p))
        )
        # pop从末尾消费：先每帧一次基础覆盖，然后消费校准加权重复。
        queues[key] = extras + base
    batches = []
    cycle = 0
    while any(queues.values()):
        batch = []
        for domain in domains:
            vids = sorted(k for k in groups if k[0] == domain)
            for j in range(batch_size // 2):
                k = vids[(cycle * (batch_size // 2) + j) % len(vids)]
                if queues[k]:
                    i = int(queues[k].pop())
                else:
                    ids = groups[k]
                    p = np.asarray(weights)[ids]
                    i = int(rng.choice(ids, p=p / p.sum()))
                batch.append(i)
        batches.append(batch)
        cycle += 1
    return batches


def exposure_report(records, batches, targets):
    counts = np.bincount(
        [i for b in batches for i in b], minlength=len(records)
    ).astype(float)
    return {
        **distribution_report(records, counts, targets),
        "observed_updates": len(batches),
        "exposure_counts": counts.astype(int).tolist(),
        "repeat_ratio": float(1 - np.count_nonzero(counts) / max(counts.sum(), 1)),
    }


def repeat_factors(records, key, target, cap=4.0):
    """尾部切片重复因子，质量只调节重复强度，不改变任务掩码。"""
    counts = Counter(str(r.get(key, "unknown")) for r in records)
    n = max(len(records), 1)
    return [
        float(
            np.clip(
                np.sqrt(
                    target.get(str(r.get(key, "unknown")), 0)
                    / max(counts[str(r.get(key, "unknown"))] / n, 1e-8)
                )
                * r.get("quality", 1),
                0,
                cap,
            )
        )
        if r.get("status", "valid") not in ("suspect", "invalid")
        else 0.0
        for r in records
    ]


def split_groups(records, ratios=(0.7, 0.15, 0.15), seed=3407):
    """以视频区段、轨迹和近重复簇构建连通分量，防止跨集合泄漏。"""
    parent = list(range(len(records)))
    seen = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, r in enumerate(records):
        domain = r["domain"]
        video = r["video"]
        segment = r.get("segment", video)
        keys = [("segment", domain, video, segment)]
        keys += [
            ("track", domain, video, segment, str(t)) for t in r.get("track_ids", [])
        ]
        if r.get("duplicate_cluster") is not None:
            keys.append(("duplicate", str(r["duplicate_cluster"])))
        for k in keys:
            if k in seen:
                parent[find(i)] = find(seen[k])
            else:
                seen[k] = i
    components = defaultdict(list)
    for i in range(len(records)):
        components[find(i)].append(i)
    rng = np.random.default_rng(seed)
    keys = list(components)
    rng.shuffle(keys)
    targets = np.asarray(ratios) * len(records)
    sizes = np.zeros(3)
    out = {}
    for k in sorted(keys, key=lambda k: -len(components[k])):
        choice = int(np.argmax(targets - sizes))
        sizes[choice] += len(components[k])
        for i in components[k]:
            out[records[i]["sample_id"]] = ("train", "calibration", "test")[choice]
    return out


def distribution_report(records, weights, targets):
    w = np.asarray(weights, float)
    total = max(w.sum(), 1e-12)
    result = {}
    for key, target in targets.items():
        values = sorted(set(str(r.get(key)) for r in records) | set(map(str, target)))
        p = {
            v: float(
                sum(w[i] for i, r in enumerate(records) if str(r.get(key)) == v) / total
            )
            for v in values
        }
        result[key] = {
            "weighted": p,
            "target": target,
            "total_variation": sum(abs(p.get(v, 0) - target.get(v, 0)) for v in values)
            / 2,
        }
    return {
        "ess": ess(w),
        "gini": gini(w),
        "max_weight": float(w.max(initial=0)),
        "marginals": result,
    }


def sinkhorn_weights(source, target, entropy=0.2, iterations=100):
    """熵正则联合表征传输；行边缘固定、目标均匀，返回耦合及搬运代价。"""
    x = np.asarray(source, float)
    y = np.asarray(target, float)
    cost = ((x[:, None] - y[None]) ** 2).sum(2)
    kernel = np.exp(-np.minimum(cost / max(entropy, 1e-8), 700))
    a = np.full(len(x), 1 / len(x))
    b = np.full(len(y), 1 / len(y))
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(iterations):
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
    plan = u[:, None] * kernel * v[None, :]
    return plan, float((plan * cost).sum())
