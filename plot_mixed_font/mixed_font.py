"""中文 / 英文 / 数学公式混排工具（matplotlib）。

为什么需要它
------------
matplotlib 只要在一个字符串里发现 ``$``，就会把**整串**交给 mathtext 数学引擎
渲染。此时字符串中的中文不再使用 ``font.family``，而是被迫用数学字体集
（如 stix）渲染——而数学字体没有中文字形，于是中文变成豆腐块 / dummy 符号。

因此「中文 + ``$...$`` 公式」**不能**写在同一个字符串里。正确做法是把字符串
按 ``$...$`` 拆成「普通文本段」和「公式段」，分别渲染后再水平拼接
（matplotlib.offsetbox.HPacker），整体作为一个对象摆放。

依赖字体
--------
- Times New Roman、SimSun（宋体）须可被 matplotlib 识别。
  本机已将字体放在 ``~/.local/share/fonts/``。新增字体后若 matplotlib 找不到，
  删除字体缓存重建即可：``rm ~/.cache/matplotlib/fontlist-*.json``。

用法
----
    from mixed_font import setup_fonts, mixed_text
    setup_fonts()
    ax.set_xticks([]); ax.set_yticks([])
    mixed_text(ax, 0.5, 0.5, r'Times New Roman和宋体: $e^{i \pi} + 1 = 0$', fontsize=30)

直接运行本文件会生成 demo 图 ``mixed_font_demo.png``。
"""
import re

import matplotlib as mpl
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

# 把字符串按 $...$ 拆开（保留公式段本身）
_MATH_SPLIT = re.compile(r'(\$[^$]*\$)')


def setup_fonts(serif='Times New Roman', cjk='SimSun', mathset='stix'):
    """配置全局字体：西文用 serif，中文用 cjk（逐字回退），公式用 mathset。"""
    mpl.rcParams['font.family'] = [serif, cjk]
    mpl.rcParams['mathtext.fontset'] = mathset


def mixed_text(ax, x, y, s, fontsize=12, ha='center', va='center',
               xycoords='axes fraction', **textprops):
    """在 ax 的 (x, y) 处绘制含中文 / 英文 / ``$...$`` 公式的混排文本。

    字符串按 ``$...$`` 自动拆分：普通文本段走常规文本路径（中文正常），
    公式段走 mathtext，再用 HPacker 水平拼接居中。

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : float        位置（坐标系由 xycoords 决定，默认 axes 比例 0~1）
    s : str             可同时包含中文、英文与 ``$...$`` 公式
    fontsize : float
    ha, va : str        水平 / 垂直对齐：'left'|'center'|'right' / 'bottom'|'center'|'top'
    xycoords : str      'axes fraction'（默认）或 'data' 等 AnnotationBbox 支持的坐标系
    **textprops         透传给每个 TextArea 的额外文本属性（如 color）

    Returns
    -------
    AnnotationBbox      已添加到 ax 的对象。
    """
    parts = [p for p in _MATH_SPLIT.split(s) if p != '']
    children = [TextArea(p, textprops=dict(fontsize=fontsize, **textprops))
                for p in parts]
    box = HPacker(children=children, align='center', pad=0, sep=0)

    box_alignment = ({'left': 0.0, 'center': 0.5, 'right': 1.0}[ha],
                     {'bottom': 0.0, 'center': 0.5, 'top': 1.0}[va])
    ab = AnnotationBbox(box, (x, y), xycoords=xycoords, frameon=False,
                        box_alignment=box_alignment)
    ax.add_artist(ab)
    return ab


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    setup_fonts()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xticks([])
    ax.set_yticks([])
    mixed_text(ax, 0.5, 0.5,
               r'Times New Roman和宋体: $e^{i \pi} + 1 = 0$',
               fontsize=28)
    fig.savefig('mixed_font_demo.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved mixed_font_demo.png')
