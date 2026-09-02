"""生成示例训练数据（演示用，跑通全流程后可替换为真实数据）。

生成 Alpaca 格式的 instruction/input/output 三元组，
覆盖通用问答、写作、代码、量化金融等领域，
输出到 data/processed/train.jsonl 与 eval.jsonl。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# 示例数据：覆盖多个领域，展示模型可被微调的方向
SAMPLES: list[dict[str, str]] = [
    # --- 通用问答 ---
    {"instruction": "用一句话解释什么是大语言模型。", "input": "", "output": "大语言模型是一种基于海量文本训练、能够理解和生成自然语言的人工智能模型。"},
    {"instruction": "什么是过拟合，如何缓解？", "input": "", "output": "过拟合指模型在训练集上表现好、在未见数据上表现差。缓解方法包括：增加数据、正则化（L1/L2、Dropout）、早停、降低模型复杂度、数据增强。"},
    # --- 写作 ---
    {"instruction": "写一段描写清晨菜市场的文字，要有烟火气。", "input": "", "output": "天刚擦亮，菜市场就醒了。卖豆腐的大婶掀开白纱布，热气混着豆香漫出来；鱼贩抄起网兜，水面被搅得哗哗响。讨价还价声、剁肉声、自行车铃铛声搅成一锅，活色生香的人间烟火。"},
    {"instruction": "把下面这句话改得更简洁有力。", "input": "我觉得这个方案从整体上来说还是有很多地方需要进一步优化和完善的。", "output": "这个方案整体上还需优化。"},
    # --- 代码 ---
    {"instruction": "用 Python 写一个函数，计算斐波那契数列第 n 项。", "input": "", "output": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
    {"instruction": "解释下面代码的作用。", "input": "df.groupby('city')['price'].mean()", "output": "按 city 列分组，计算每个城市的 price 列平均值。"},
    # --- 量化金融 ---
    {"instruction": "什么是均线（MA），金叉和死叉代表什么？", "input": "", "output": "均线是过去 N 日收盘价的平均值连线，反映趋势。短期均线上穿长期均线称金叉，通常视为买入信号；反之称死叉，通常视为卖出信号。"},
    {"instruction": "解释资金费率对合约交易的影响。", "input": "", "output": "资金费率是多空双方周期性互相支付的费用，用于让永续合约价格贴近现货。费率为正时多方付费给空方，市场偏多；为负则相反。高费率往往伴随过度投机。"},
    # --- 学习 / 方法论 ---
    {"instruction": "如何高效学习一门新编程语言？", "input": "", "output": "1）明确目标：想用它做什么；2）学最小语法后立刻写小项目；3）读优秀开源代码；4）遇到问题查文档而非死记；5）持续输出、复盘。"},
    {"instruction": "给我一个坚持长期写作的方法。", "input": "", "output": "把写作拆成可执行的最小动作：每天固定时间写 300 字，降低启动门槛；建立素材库随时记录灵感；先完成再完美；公开输出借助外部反馈推动自己。"},
]


def make_samples(
    n_train: int = 50,
    n_eval: int = 10,
    out_dir: Path = Path("data/processed"),
    seed: int = 42,
) -> None:
    """基于模板扩增生成指定数量的训练/验证样本。"""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 简单扩增：对样本做轻微改写占位，实际场景应替换为真实多样数据
    def expand(base: list[dict], count: int) -> list[dict]:
        result: list[dict] = []
        while len(result) < count:
            for item in base:
                if len(result) >= count:
                    break
                result.append(dict(item))
        return result

    train = expand(SAMPLES, n_train)
    eval_set = expand(SAMPLES, n_eval)

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write(out_dir / "train.jsonl", train)
    write(out_dir / "eval.jsonl", eval_set)

    print(f"✅ 已生成示例数据：{out_dir/'train.jsonl'} ({len(train)} 条)")
    print(f"✅ 已生成示例数据：{out_dir/'eval.jsonl'} ({len(eval_set)} 条)")
    print("⚠️  这是演示数据，正式训练请替换为你的真实数据（见 docs/data.md）")


if __name__ == "__main__":
    make_samples()
