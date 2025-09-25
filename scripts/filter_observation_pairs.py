import json
import re
from pathlib import Path

# 使路径与脚本位置无关：脚本位于 project_root/codes/abstract-visual-token/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / 'new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_9.23_further_washed.json'
DST = PROJECT_ROOT / 'new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_9.24_further_washed.json'

# Pattern to find <observation>...</observation> or detect unclosed tags
OPEN_TAG = '<observation>'
CLOSE_TAG = '</observation>'

# Regex to capture properly closed segments (non-greedy)
PAIR_RE = re.compile(r'<observation>.*?</observation>', re.DOTALL)
# Regex to detect any tag fragments
ANY_TAG_RE = re.compile(r'</?observation>')

def assistant_texts(sample):
    for turn in sample.get('data', []):
        if turn.get('role') == 'assistant':
            for c in turn.get('content', []):
                if c.get('type') == 'text':
                    yield c.get('text', '')

def is_sample_valid(sample):
    # Collect all assistant text concatenated (or per segment check?)
    # We validate per text segment: each segment either
    #  1) contains no observation tag at all, or
    #  2) all observation tags in that segment are properly paired (every open has a close and order correct)
    for txt in assistant_texts(sample):
        if OPEN_TAG in txt or CLOSE_TAG in txt:
            # Quick stack-based validation for nested/ordering (though nesting probably not expected)
            stack = 0
            i = 0
            while i < len(txt):
                if txt.startswith(OPEN_TAG, i):
                    stack += 1
                    i += len(OPEN_TAG)
                elif txt.startswith(CLOSE_TAG, i):
                    stack -= 1
                    if stack < 0:
                        return False
                    i += len(CLOSE_TAG)
                else:
                    i += 1
            if stack != 0:
                return False
    return True

def main():
    if not SRC.exists():
        raise SystemExit(f'Source file not found: {SRC}')

    with SRC.open('r', encoding='utf-8') as f:
        data = json.load(f)

    kept = []
    dropped = 0

    for sample in data:
        if is_sample_valid(sample):
            # Additionally ensure at least one properly closed pair if any tag appears
            any_tag = any((OPEN_TAG in t or CLOSE_TAG in t) for t in assistant_texts(sample))
            if any_tag:
                # Optional: verify every close/open is in pairs (already by is_sample_valid)
                kept.append(sample)
            else:
                # If无<observation>需求? 题意: 判断"assistant"content中的<observation>是否能匹配</observation>。若某一<observation>缺失</observation>则扔掉。无标签样本视为不含缺失 => 保留
                kept.append(sample)
        else:
            dropped += 1

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open('w', encoding='utf-8') as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f'Total: {len(data)}')
    print(f'Kept: {len(kept)}')
    print(f'Dropped: {dropped}')

if __name__ == '__main__':
    main()
