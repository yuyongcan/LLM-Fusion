"""冒烟测试：五个域的 reward scorer 是否可用（依赖是否装齐）。

必须作为文件运行（不能用 python - <<EOF），因为 science/math verifier 用
multiprocessing spawn，子进程会去 import __main__ 的文件路径。
"""

from verl.utils.reward_score import default_compute_score as dcs

CASES = [
    # (data_source, solution_str, ground_truth)
    ("aime2025", r"so the answer is \boxed{42}", "42"),
    ("aime2025", r"so the answer is \boxed{41}", "42"),
    ("gpqa", r"\boxed{A}", "A"),
    ("ifeval", "hello world", None),
    ("ifbench", "hello world", None),
]


def main():
    for src, sol, gt in CASES:
        try:
            if gt is None:
                # IF 需要 extra_info 里的 constraint 列表，单独在下面测
                continue
            print(f"{src:12s} gt={gt!r:6s} -> {dcs(src, sol, gt)}")
        except Exception as e:  # noqa: BLE001
            print(f"{src:12s} FAIL {type(e).__name__}: {e}")

    # IF 域：直接测底层 verifier 的依赖能否 import
    try:
        import emoji  # noqa: F401
        import immutabledict  # noqa: F401
        import langdetect  # noqa: F401
        import nltk
        import syllapy  # noqa: F401

        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        print("IF deps      -> OK (nltk corpora present)")
    except Exception as e:  # noqa: BLE001
        print(f"IF deps      FAIL {type(e).__name__}: {e}")

    # math-verify 的 antlr parser extra
    try:
        import antlr4

        print(f"antlr4       -> OK {getattr(antlr4, '__version__', 'n/a')}")
    except Exception as e:  # noqa: BLE001
        print(f"antlr4       FAIL {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
