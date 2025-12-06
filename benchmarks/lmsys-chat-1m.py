import time
import sglang as sgl
from datasets import load_dataset
import pandas as pd
import numpy as np
from sglang.utils import terminate_process

# ================= 配置区域 =================
# 模型路径 (请替换为您本地的模型路径或 HuggingFace ID)
MODEL_PATH = "/HOME/nju_mli/nju_mli_1/HDD_POOL/dpskv2"
DATASET_PATH = "/HOME/nju_mli/nju_mli_1/junjie/RadixRetro/datasets/LMSYS-Chat-1M"

# 采样对话数量 (为了快速测试，默认只跑前 10 个对话)
NUM_SAMPLES = 10

# 是否禁用 RadixCache (设置为 True 可以对比性能差异)
DISABLE_RADIX = False
# ===========================================


def load_lmsys_data(num_samples=10):
    """
    加载 LMSYS-Chat-1M 数据集并进行预处理。
    由于数据集较大，我们使用流式加载或只取前几条。
    """
    print(f"正在加载 LMSYS-Chat-1M 数据集 (前 {num_samples} 条)...")
    try:
        # 使用流式加载以节省内存
        dataset = load_dataset(DATASET_PATH, split="train", streaming=True)
        data_iter = iter(dataset)

        conversations = []
        for _ in range(num_samples):
            try:
                item = next(data_iter)
                # 提取 conversation 字段
                conversations.append(item["conversation"])
            except StopIteration:
                break
        return conversations
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("将使用模拟的多轮对话数据进行测试。")
        # 模拟数据：如果不方便下载数据集，使用此数据
        dummy_convo = [
            {"role": "user", "content": "你好，请介绍一下你自己。"},
            {"role": "assistant", "content": "我是由人工智能技术构建的大型语言模型。"},
            {"role": "user", "content": "你会写Python代码吗？"},
            {"role": "assistant", "content": "是的，我会写Python代码。"},
            {"role": "user", "content": "那请给我写一个快速排序算法。"},
            {"role": "assistant", "content": "好的，这是快速排序的代码..."},
            {"role": "user", "content": "解释一下这段代码的时间复杂度。"},
        ]
        return [dummy_convo] * num_samples


@sgl.function
def chat_inference(s, history_prompt):
    """
    SGLang 的推理函数。
    注意：为了利用 RadixCache，我们只把 Prompt 传进来，
    SGLang 引擎会自动匹配这段 Prompt 是否在 Cache 中存在。
    """
    s += history_prompt
    s += sgl.gen("response", max_tokens=128, stop=["<|eot_id|>", "user:"])


def main():
    # 1. 初始化 SGLang Runtime (离线模式)
    print(
        f"正在初始化 SGLang Runtime (RadixCache={'Disabled' if DISABLE_RADIX else 'Enabled'})..."
    )

    # mem_fraction_static 是 KV Cache 的显存占比，调大可以缓存更多历史
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        disable_radix_cache=DISABLE_RADIX,
        mem_fraction_static=0.8,
    )
    sgl.set_default_backend(runtime)

    # 2. 准备数据
    conversations = load_lmsys_data(NUM_SAMPLES)

    total_latency = 0
    total_turns = 0
    latencies = []

    print("\nStarting Benchmark Loop...")
    print("-" * 60)

    # 3. 循环对话
    for i, convo in enumerate(conversations):
        # 这是一个对话 Session
        history = ""
        print(f"\nProcessing Conversation {i + 1}/{len(conversations)}")

        for turn_idx, msg in enumerate(convo):
            # 我们只模拟用户发给模型的情况
            if msg["role"] == "user":
                # 构造符合 Chat 模板的 Prompt (这里做简化处理，实际需适配模型模板)
                # 关键：history 包含之前的所有内容，这部分应当命中 Cache
                current_prompt = f"User: {msg['content']}\nAssistant:"
                full_input = history + current_prompt

                start_time = time.time()

                # 调用 SGLang 进行生成
                state = chat_inference.run(history_prompt=full_input)
                generated_text = state["response"]

                end_time = time.time()
                latency = end_time - start_time

                # 更新历史，模拟多轮对话的积累
                history += current_prompt + generated_text + "\n"

                latencies.append(latency)
                total_turns += 1

                print(
                    f"  Turn {turn_idx // 2 + 1}: Latency = {latency:.4f}s | Input Len: {len(full_input)} chars"
                )

    # 4. 统计结果
    trace_meta = {
        "benchmark": "lmsys-chat-1m",
        "num_samples": NUM_SAMPLES,
        "total_turns": total_turns,
    }
    try:
        dump_resp = runtime.dump_radix_tree(
            path="tree_node_trace.json",
            meta=trace_meta,
        )
        trace_path = (
            dump_resp["path"]
            if isinstance(dump_resp, dict) and "path" in dump_resp
            else "tree_node_trace.json"
        )
        print(f"RadixTree trace dumped to {trace_path}")
    except Exception as e:
        print(f"Failed to dump RadixTree trace: {e}")

    print("-" * 60)
    print(f"Benchmark 完成。")
    print(f"平均每轮延迟: {np.mean(latencies):.4f} s")
    print(f"P99 延迟: {np.percentile(latencies, 99):.4f} s")

    if not DISABLE_RADIX:
        print("\n[分析] 如果 RadixCache 生效，你会发现随着 Input Len 变长，")
        print("       延迟并没有线性大幅增加，因为前缀部分（History）被跳过了计算。")

    # 关闭 Runtime
    runtime.shutdown()


if __name__ == "__main__":
    main()
