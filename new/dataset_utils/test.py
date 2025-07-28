import re
def clean_problem_image(txt: str) -> str:
    # (?s) = DOTALL，让 . 匹配换行
    pattern = (
        r'(?s)(?<=\.\s{1})'   # 可变长向后断言：定位在 ".␠␠" 之后
        r'.*?:\n'             # 非贪婪地吃掉直到最近 ":\n"（含它）
        r'<image_start>\[problem_image_1\]<image_end>'  # 图片标签
    )
    txt = re.sub(pattern, '', txt)  # 直接整段清空
    txt = re.sub(r'<image_start>\[problem_image_1\]<image_end>', '', txt)
    return txt

s = "The vertices of a convex pentagon are P1=(-1, -1), P2=(-3, 4), P3=(1, 7), P4=(6, 5), and P5=(3, -1).\n<image_start>[problem_image_1]<image_end>\nWhat is the area of the pentagon?"


print(clean_problem_image(s))