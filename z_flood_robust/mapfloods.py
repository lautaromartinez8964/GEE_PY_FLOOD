# 该文件结合Sentinel-1后向散射数据，JRC全球地表水数据集，以及可选的DSWE(Dynamic Surface Water Extent)数据，来检测洪水并生成洪水分类图

# 导入模块
import ee
from .zscore import *
import warnings

try:
    from eedswe import cdswe

    has_eedswe = True
except ImportError:
    has_eedswe = False
"""
mapFloods函数
函数参数：
  z:输入的Sentinel-1图像（ee.Image对象）
  zvv_thd:vv波段的Z-score阈值。小于该阈值即认为是洪水
  zvh_thd:VH波段的z-score阈值
  pow_thd:永久性开放水域的阈值概率，默认值为90
  pin_thd:季节性淹没的概率阈值，默认值为25
  use_dswe:是否使用DSWE算法，若使用需安装eedswe包
  dswe_start 和 dswe_end:DSWE方法的起始与结束日期
  doy_start 和 doy_end:DSWE的起始和结束日期占一年中的第几天

返回值：返回一个ee.Image对象，表示洪水分类图
"""


def mapFloods(
        z,
        zvv_thd,
        zvh_thd,
        pow_thd=90,
        pin_thd=25,
        use_dswe=False,
        dswe_start="2000-01-01",
        dswe_end="2018-01-01",
        doy_start=1,
        doy_end=366
):
    jrc = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")

    def _getvalid(x):
        return x.gt(0)

    # jrcvalid:计算有效数据(像素值大于0）的总和
    jrcvalid = jrc.map(_getvalid).sum()
    jrcmask = jrcvalid.gt(0)

    def _getwat(x):
        return x.eq(2)

    # jrcwat:计算开放水域的频率（百分比）,即基线期jrc数据中某像素等于2的概率（像素值等于2（开放水体）/像素值大于0（有效数据）
    jrcwat = jrc.map(_getwat).sum().divide(jrcvalid).multiply(100)
    # ow:根据pow_thd阈值生成永久开放水域（历史水体）掩码
    ow = jrcwat.gt(pow_thd)

    # 如果使用DSWE算法
    if use_dswe:
        # 先定义DSWE数据的过滤条件
        dswe_filters = [
            ee.Filter.date(dswe_start, dswe_end),
            ee.Filter.dayOfYear(doy_start, doy_end)
        ]
        # 调用cdswe函数，获取符合过滤条件的DSWE数据 pdswe为DSWE数据集，包含多个波段如pDSWE1、pDSWE2等
        pdswe = cdswe(dswe_filters)
        # 获取历史永久性开放水体 判断pDSWE1波段（高概率水体概率）是否大于阈值，大于即为永久水体
        ow = ow.where(pdswe.select("pDSWE1").gte(ee.Image(pow_thd)), 1)
        # 计算季节性淹没概率：高，中，低水体概率相加的总概率与阈值比较
        pinun = pdswe.select("pDSWE1").add(pdswe.select("pDSWE2")).add(pdswe.select("pDSWE3"))
        inun = pinun.gte(pin_thd)
    else:
        inun = jrcwat.gte(pin_thd)

    # 生成洪水分类图 返回VV,VH波段的洪水图
    vvflag = z.select('VV').lte(zvv_thd)
    vhflag = z.select('VH').lte(zvh_thd)

    # 生成洪水分类图：根据多个二值图像，对不同方法所得洪水赋予不同的加值最后返回一张ee.Image的洪水类别图
    """
    vvflag:若是vv洪水，像素值增加1
    vhflag：若是vh洪水，像素值增加2
    inun:若是季节性淹没洪水，像素值增加10
    ow:若是永久性开放水域，则将像素值设置为20
    """
    flood_class = ee.Image(0) \
        .add(vvflag)\
        .add(vhflag.multiply(2))\
        .add(inun.multiply(10))\
        .where(ow.eq(1),20)\
        .rename('flood_class')\
        .updateMask(jrcmask)
    """
     0：非水域；非洪水。

    1：仅 VV 波段洪水标志。

    2：仅 VH 波段洪水标志。

    3：VV 和 VH 波段洪水标志。

    10：历史季节性淹没区域；无洪水标志。

    11：历史季节性淹没区域；仅 VV 波段洪水标志。

    12：历史季节性淹没区域；仅 VH 波段洪水标志。

    13：历史季节性淹没区域；VV 和 VH 波段洪水标志。

    20：永久性开放水域。
    """
    return flood_class


floodPalette = [
    '#000000',  # 0 - non-water; non-flood
    '#FC9272',  # 1 - VV only
    '#FC9272',  # 2 - VH only
    '#FF0000',  # 3 - VV + VH
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#DEEBF7',  # 10 - prior inundation; no flag
    '#8C6BB1',  # 11 - prior inundation; VV only
    '#8C6BB1',  # 12 - prior inundation; VH only
    '#810F7C',  # 13 - prior inundation; VV + VH
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#08306B'  # 20 - permanent open water
]