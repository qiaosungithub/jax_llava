import utils.state_util as state_util
from utils.logging_util import log_for_0, print0


# Function to print number of parameters
def print_params(params):
    params_flatten = state_util.flatten_state_dict(params)

    total_params = 0
    max_length = max(len(k) for k in params_flatten.keys())
    
    def _get_shape(p):
        if hasattr(p, 'shape'):
            return p.shape
        elif hasattr(p, 'value'):
            return p.value.shape
        else:
            return "unknown"
        
    def _get_size(p):
        if hasattr(p, 'size'):
            return p.size
        elif hasattr(p, 'value'):
            return p.value.size
        else:
            return "unknown"
    
    max_shape = max(len(f"{_get_shape(p)}") for p in params_flatten.values())
    max_digits = max(len(f"{_get_size(p):,}") for p in params_flatten.values())
    print0('-' * (max_length + max_digits + max_shape + 8), flush=True)
    for name, param in params_flatten.items():
        layer_params = _get_size(param)
        str_layer_shape = f"{_get_shape(param)}".rjust(max_shape) 
        str_layer_params = f"{layer_params:,}".rjust(max_digits)
        print0(f" {name.ljust(max_length)} | {str_layer_shape} | {str_layer_params} ", flush=True)
        total_params += layer_params
    print0('-' * (max_length + max_digits + max_shape + 8), flush=True)
import re
from collections import OrderedDict, defaultdict

def _format_indices(indices):
    idx = sorted(set(indices))
    if not idx:
        raise ValueError("Empty indices in _format_indices()")

    parts = []
    start = prev = idx[0]

    def _flush_run(a, b):
        run_len = b - a + 1
        if run_len >= 4:
            parts.append(f"{a}...{b}")
        else:
            parts.extend(str(x) for x in range(a, b + 1))

    for x in idx[1:]:
        if x == prev + 1:
            prev = x
        else:
            _flush_run(start, prev)
            start = prev = x
    _flush_run(start, prev)

    return "{" + ",".join(parts) + "}"


def _prod_shape(shape):
    p = 1
    for d in shape:
        if not isinstance(d, int):
            raise TypeError(f"Non-int dim in shape: {shape}")
        p *= d
    return p


def count_params(params):
    params_flatten = state_util.flatten_state_dict(params)

    total = 0
    for name, p in params_flatten.items():
        if hasattr(p, "shape"):
            shape = tuple(p.shape)
        elif hasattr(p, "value") and hasattr(p.value, "shape"):
            shape = tuple(p.value.shape)
        else:
            raise TypeError(f"Param has no shape: {name}")

        if hasattr(p, "size"):
            size = int(p.size)
        elif hasattr(p, "value") and hasattr(p.value, "size"):
            size = int(p.value.size)
        else:
            raise TypeError(f"Param has no size: {name}")

        if _prod_shape(shape) != size:
            raise ValueError(f"shape/size mismatch: {name} shape={shape} size={size}")

        total += size

    return total


def print_params_compact(params):
    params_flatten = state_util.flatten_state_dict(params)

    # 匹配路径段：/layers_7 或 /layer_7；只替换第一个
    first_layer_re = re.compile(r'(^|/)(layers|layer)_(\d+)(?=/|$)')
    # 用于把 normalized key 里的 /layers_*/ 替换成 /layers_{...}/
    first_layer_star_re = re.compile(r'(^|/)(layers|layer)_\*(?=/|$)')

    def _get_shape_strict(name, p):
        if hasattr(p, "shape"):
            return tuple(p.shape)
        if hasattr(p, "value") and hasattr(p.value, "shape"):
            return tuple(p.value.shape)
        raise TypeError(f"Param has no shape: {name}")

    def _get_size_strict(name, p):
        if hasattr(p, "size"):
            return int(p.size)
        if hasattr(p, "value") and hasattr(p.value, "size"):
            return int(p.value.size)
        raise TypeError(f"Param has no size: {name}")

    order = []                 # ('single', name) or ('group', normalized_name)
    singles = OrderedDict()    # name -> entry
    groups = OrderedDict()     # normalized_name -> dict(entries=list[entry], seen_idx=set[int])
    seen_group = set()

    all_names = set(params_flatten.keys())

    for pos, (name, p) in enumerate(params_flatten.items()):
        shape = _get_shape_strict(name, p)
        size = _get_size_strict(name, p)

        if _prod_shape(shape) != size:
            raise ValueError(f"shape/size mismatch: {name} shape={shape} size={size}")

        m = first_layer_re.search(name)
        if m:
            layer_idx = int(m.group(3))

            # 把第一处层号归一化为 *_ 作为 group key
            normalized = first_layer_re.sub(
                lambda mm: f"{mm.group(1)}{mm.group(2)}_*",
                name,
                count=1
            )

            g = groups.get(normalized)
            if g is None:
                groups[normalized] = g = {"entries": [], "seen_idx": set()}

            if layer_idx in g["seen_idx"]:
                raise ValueError(f"Duplicate first-layer index {layer_idx} in group: {normalized}")
            g["seen_idx"].add(layer_idx)

            g["entries"].append({
                "pos": pos,
                "name": name,
                "idx": layer_idx,
                "shape": shape,
                "size": size,
            })

            if normalized not in seen_group:
                order.append(("group", normalized))
                seen_group.add(normalized)
        else:
            singles[name] = {"pos": pos, "name": name, "shape": shape, "size": size}
            order.append(("single", name))

    # rows: (name_out, shape_out, total_str, detail_str, anchor_pos)
    rows = []
    printed_sum = 0
    used_names = set()

    for kind, key in order:
        if kind == "single":
            e = singles[key]
            total = e["size"]
            total_str = f"{total:,}"
            # 如果你更想要单个行 detail 为空，把下一行改成 detail_str = ""
            detail_str = f"{e['size']:,}"
            rows.append((e["name"], e["shape"], total_str, detail_str, e["pos"]))

            printed_sum += total
            if e["name"] in used_names:
                raise AssertionError(f"Param printed twice: {e['name']}")
            used_names.add(e["name"])
            continue

        if kind != "group":
            raise RuntimeError(f"Unknown order kind: {kind}")

        g = groups[key]

        buckets = defaultdict(list)  # (size, shape) -> [entry...]
        for e in g["entries"]:
            buckets[(e["size"], e["shape"])].append(e)

        group_lines = []
        for (size, shape), ents in buckets.items():
            ents_sorted = sorted(ents, key=lambda x: x["pos"])
            anchor = ents_sorted[0]["pos"]

            if len(ents_sorted) >= 2:
                occurs = len(ents_sorted)
                idxs = [e["idx"] for e in ents_sorted]
                idx_str = _format_indices(idxs)  # '{0,1}' or '{0...3,5,...}'
                total = size * occurs

                # 关键：把路径里的 layers_* 替换成 layers_{...}
                def _repl(mm):
                    prefix, token = mm.group(1), mm.group(2)
                    return f"{prefix}{token}_{idx_str}"

                name_out = first_layer_star_re.sub(_repl, key, count=1)

                total_str = f"{total:,}"
                detail_str = f"{occurs} * {size:,}"  # ✅ 你要的格式
                group_lines.append((name_out, shape, total_str, detail_str, anchor))

                printed_sum += total
                for e in ents_sorted:
                    if e["name"] in used_names:
                        raise AssertionError(f"Param printed twice: {e['name']}")
                    used_names.add(e["name"])
            else:
                e = ents_sorted[0]
                total = e["size"]
                total_str = f"{total:,}"
                detail_str = f"{e['size']:,}"  # 或 ""（按你喜好）
                group_lines.append((e["name"], e["shape"], total_str, detail_str, anchor))

                printed_sum += total
                if e["name"] in used_names:
                    raise AssertionError(f"Param printed twice: {e['name']}")
                used_names.add(e["name"])

        group_lines.sort(key=lambda x: x[4])  # anchor_pos
        rows.extend(group_lines)

    # 严格覆盖与总和校验
    if used_names != all_names:
        missing = sorted(all_names - used_names)
        extra = sorted(used_names - all_names)
        raise AssertionError(f"Printed coverage mismatch. missing={missing[:5]} extra={extra[:5]}")

    expected_total = count_params(params)
    if printed_sum != expected_total:
        raise AssertionError(f"Total mismatch: printed_sum={printed_sum:,} expected_total={expected_total:,}")

    # 打印对齐：四列（name | shape | total | detail）
    max_name = max(len(r[0]) for r in rows) if rows else 1
    max_shape = max(len(str(r[1])) for r in rows) if rows else 1
    max_total = max(len(r[2]) for r in rows) if rows else 1
    max_detail = max(len(r[3]) for r in rows) if rows else 1

    line = "-" * (max_name + max_shape + max_total + max_detail + 12)
    print0(line, flush=True)
    for name_out, shape_out, total_str, detail_str, _ in rows:
        print0(
            f" {name_out.ljust(max_name)} | "
            f"{str(shape_out).rjust(max_shape)} | "
            f"{total_str.rjust(max_total)} | "
            f"{detail_str.rjust(max_detail)} ",
            flush=True
        )
    print0(line, flush=True)
    print0(f" Total params: {expected_total:,}", flush=True)

def print_param_shapes(param_shapes):
    params_flatten = state_util.flatten_state_dict(param_shapes)

    max_length = max(len(k) for k in params_flatten.keys())
    max_shape = max(len(f"{p}") for p in params_flatten.values())
    print0('-' * (max_length + max_shape + 5), flush=True)
    for name, shape in params_flatten.items():
        str_layer_shape = f"{shape}".rjust(max_shape) 
        print0(f" {name.ljust(max_length)} | {str_layer_shape} ", flush=True)
    print0('-' * (max_length + max_shape + 5), flush=True)