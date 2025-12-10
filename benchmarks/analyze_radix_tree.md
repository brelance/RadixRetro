# analyze_radix_tree.py 使用说明

该脚本用于对 `tree_node_trace.json` 中的 Radix Tree 节点进行离线分析，可一次性生成两份互不干扰的统计 JSON，帮助你快速了解节点数量、子树分布以及热点命中情况。

## 核心功能
- **节点总览**：统计节点总数以及所有节点 `tokens` 的长度总和。
- **多子节点分析**：找出 `children_count > 1` 的节点，并记录其 `node_id`、`hit_count`、子节点数量以及 `tokens_length`。
- **命中热点排序**：按照 `hit_count` 进行降序排序，输出所有节点的 `node_id`、`tokens`、`hit_count` 以及 `tokens_length` 方便查找高频节点。

## 环境准备
- Python ≥ 3.8（推荐直接使用 `python3` 命令）。
- 在运行目录下准备好目标 `tree_node_trace.json` 文件；默认脚本会读取当前仓库根目录下的文件。

## 快速开始
```bash
# 在仓库根目录执行，读取默认的 tree_node_trace.json
python3 benchmarks/analyze_radix_tree.py
```
运行结束后会看到类似输出：
```
Total nodes: 65. Nodes with >1 child: 4. Details saved to benchmarks/radix_tree_multi_child_nodes.json.
```

## 命令行参数
| 参数 | 默认值 | 作用 |
| ---- | ------ | ---- |
| `--input` | `tree_node_trace.json` | 指定要分析的快照文件路径。 |
| `--output` | `benchmarks/radix_tree_multi_child_nodes.json` | 多子节点统计结果输出路径。 |
| `--hits-output` | `benchmarks/radix_tree_nodes_by_hit.json` | 按 `hit_count` 排序的节点列表输出路径。 |

所有参数都接受绝对路径或相对路径，你可以根据需要将结果写入其他目录，例如：
```bash
python3 benchmarks/analyze_radix_tree.py \
  --input traces/tree_node_trace_90.json \
  --output traces/results/multi_child.json \
  --hits-output traces/results/nodes_by_hit.json
```

## 输出文件
1. **多子节点统计 (`--output`)**
   ```json
   {
     "source": "tree_node_trace.json",
     "total_nodes": 65,
     "total_tokens_length": 5228,
     "multi_child_nodes_count": 4,
     "multi_child_tokens_length": 400,
     "multi_child_nodes": [
       {
         "node_id": 4,
         "hit_count": 59,
         "tokens_length": 1,
         "children_count": 2
       }
     ]
   }
   ```
   - 包含整体节点信息与所有 `children_count > 1` 的节点数组。

2. **按命中数排序 (`--hits-output`)**
   ```json
   [
     {
       "node_id": 4,
       "tokens": [100000],
       "hit_count": 59,
       "tokens_length": 1
     },
     ...
   ]
   ```
   - 所有节点均会保存，按 `hit_count` 降序排列，便于查找热点。

## 常见用法
- **批量分析不同快照**：结合 `benchmarks/run_analyze_radix_tree.sh` 循环设置 `TRACE_ID`，即可一键生成多个快照的统计文件。
- **排查热点节点**：优先查看 `--hits-output` 结果顶部节点，确认是否需要扩大缓存或优化路径。
- **多子树结构排查**：使用 `--output` 中的 `multi_child_nodes` 列表定位分叉点，再配合 `visualize_radix_tree.py` 进行图形化调试。

## 故障排查
- **找不到输入文件**：确认 `--input` 参数路径正确，或将快照文件复制到仓库根目录并沿用默认配置。
- **权限问题**：输出路径位于受限目录时，请先创建目录或选择当前仓库下的可写路径。
- **Python 版本冲突**：在同时安装多套 Python 时，显式使用 `python3` 或虚拟环境中的解释器。

完成以上步骤后，你即可高效复用 `analyze_radix_tree.py` 的所有分析能力。祝调试顺利！
