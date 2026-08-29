# Fonts

The figures in `assets/` are drawn with HarmonyOS Sans SC. The three files here
are Latin-only subsets, instanced out of the variable original at weights 400,
600 and 700 by `../make_fonts.py`. Together they are about 130 KB, against
20 MB for the source, and they let `../figstyle.py` render a figure without
asking anything of the machine it runs on.

HarmonyOS Sans is Huawei's, free to use and redistribute. Source and license:
<https://developer.huawei.com/consumer/cn/design/resource/>

Nothing in `mamba3.py` touches these files.
