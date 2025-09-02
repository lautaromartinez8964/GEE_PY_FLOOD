"""
该份代码用于计算时间序列中的基线图像统计量（中值，mad）以及基于这些统计量的异常值和Robust-Z分数
"""

import ee


# 第一部分：过滤函数

# 1.根据卫星的轨道方向(ASCENDING或DESCENDING)过滤图像几何（输入图像为x）
def filterOrbit(x, direction):
    return x.filter(ee.Filter.equals('orbitProperties_pass', direction))


# 2.根据Sentinel-1的成像模式(IW,SM)过滤图像集合
def filterMode(x, mode):
    return x.filter(ee.Filter.equals('instrumentMode', mode))


# 第二部分:计算各统计值

# 1.计算基线期（设定为非洪水期）Sentinel-1图像集合的均值，给定日期

"""
该函数参数：
  x：输入的Sentinel-1图像集合
  start:基线期的开始日期（格式为YYYY-MM-DD)
  end:基线期的结束日期
  mode、direction：成像模式和轨道方向
返回：一个ee.Image单图像，代表了基线期内区域的后向散射平均值
"""


def calc_basemean(x, start, end, mode="IW", direction="DESCENDING"):
    return x \
        .filter(ee.Filter.equals('orbitProperties_pass', direction)) \
        .filter(ee.Filter.equals('instrumentMode', mode)) \
        .filterDate(start, end) \
        .mean()


# 新增：计算基线期中位数 z-robust
"""
计算基线期 Sentinel-1 图像集合的中位数。

    Args:
      x: 输入的 ee.ImageCollection (应包含指定 bands)。
      start: 基线期的开始日期 (格式为 YYYY-MM-DD)。
      end: 基线期的结束日期。
      mode: 成像模式 ("IW" 或 "SM")。
      direction: 轨道方向 ("ASCENDING" 或 "DESCENDING")。
      

    Returns:
      一个 ee.Image 对象，代表基线期内区域指定波段的中位数。
"""


def calc_basemedian(x, start, end, mode="IW", direction="DESCENDING"):
    baseline_collection = x \
        .filter(ee.Filter.equals('orbitProperties_pass', direction)) \
        .filter(ee.Filter.equals('instrumentMode', mode)) \
        .filterDate(start, end)
    return baseline_collection.median()


# 新增:计算基线期绝对离差中位数(MAD)
"""
  Args:
      x: 输入的 ee.ImageCollection (应包含指定 bands)。
      start, end, mode, direction: 同上。
      

    Returns:
      一个 ee.Image 对象，代表基线期内区域指定波段的 MAD。
      如果计算出的 MAD 为 0，会被替换为一个极小值 (epsilon) 以防止除零错误。

"""


def calc_basemad(x, start, end, mode="IW", direction="DESCENDING"):
    basemedian = calc_basemedian(x, start, end, mode, direction)  # 1.计算基线中位数
    baseline_collection = x \
        .filter(ee.Filter.equals('orbitProperties_pass', direction)) \
        .filter(ee.Filter.equals('instrumentMode', mode)) \
        .filterDate(start, end)  # 2.获取基线影像集合

    # 3.计算每个影像与中位数的绝对差值
    def calc_abs_diff(image):
        # 确保只减去对应波段的中位数
        median_for_image = basemedian.select(image.bandNames())
        return image.subtract(median_for_image).abs() \
            .set({'system:time_start': image.get('system:time_start')})

    abs_diff_collection = baseline_collection.map(calc_abs_diff)

    # 4.计算绝对差值的中位数(即MAD）
    basemad = abs_diff_collection.median()

    # 5.处理MAD可能为0的情况 如果基线期某个像素值完全没变，MAD会是0，导致后续除0.将MAD为0的像素替换为一个非常小的正数
    epsilon = 1e-6
    basemad_safe = basemad.where(basemad.eq(0), epsilon)

    return basemad_safe


# 2.计算基线期标准差图像

"""
该函数参数：
  x：输入的Sentinel-1图像集合
  start:基线期的开始日期（格式为YYYY-MM-DD)
  end:基线期的结束日期
返回：一个ee.Image单图像，表示基线期内的标准差图像
"""


def calc_basestd(x, start, end, mode="IW", direction="DESCENDING"):
    return x \
        .filter(ee.Filter.equals('orbitProperties_pass', direction)) \
        .filter(ee.Filter.equals('instrumentMode', mode)) \
        .filterDate(start, end) \
        .reduce(ee.Reducer.stdDev()) \
        .rename(['VV', 'VH', 'angle'])  # VV,VH和入射角波段都计算


# 新增：计算每幅图像中位数异常
def calc_median_anomaly(x, start, end, mode="IW", direction="DESCENDING"):
    """
    计算每幅图像相对于基线中位数的差异

    """
    basemedian = calc_basemedian(x, start, end, mode, direction)

    def _calc_median_anom(y):
        median_for_y = basemedian.select(y.bandNames())
        return y.subtract(median_for_y).set({'system:time_start': y.get('system:time_start')})

    # 应用到过滤的集合上
    anomaly_collection = x \
        .filter(ee.Filter.equals('orbitProperties_pass', direction)) \
        .filter(ee.Filter.equals('instrumentMode', mode)) \
        .map(_calc_median_anom)  # 计算应用异常的函数

    return anomaly_collection


# 3.计算每幅图像异常值

"""
参数同上
返回值：返回一个ee.ImageCollection图像集合，每张图像表示原本图像的异常值，即每张图像与基线均值的差异
细节：·首先计算基线均值basemean
     ·定义一个内部函数_calcanom(y),用于计算每个图象与基线均值的差异，并保留时间戳
      ·使用map方法将_calcanom应用到每个图象上
       
"""


def calc_anomaly(x, start, end, mode="IW", direction="DESCENDING"):
    basemean = calc_basemean(x, start, end, mode, direction)

    def _calcanom(y):
        return y \
            .subtract(basemean) \
            .set({'system:time_start': y.get('system:time_start')})

    return x \
        .filter(ee.Filter.equals('orbitProperties_pass', direction)) \
        .filter(ee.Filter.equals('instrumentMode', mode)) \
        .map(_calcanom)


# 4.计算每幅图像Z分数
#  Z = (X-u）/σ

def calc_zscore(x, start, end, mode="IW", direction="DESCENDING"):
    '''
    Computes the pixelwise backscatter Z-scores for each image in a collection, given a baseline period, acquisition mode and orbital direction

    Args:
    =====
    x:          A Sentinel-1 ee.ImageCollection
    start:      Start date of baseline period ("YYYY-MM-DD")
    end:        End date of baseline period ("YYYY-MM-DD")
    mode:       Acquisition mode. Can be one of "IW" (default) or "SM"
    direction:  Orbital direction. Can be either "DESCENDING" (default) or "ASCENDING"

    Returns:
    ========
    An ee.ImageCollection object that represents the pixelwise backscatter Z-scores for each image in the input ImageCollection
    '''
    anom = calc_anomaly(x, start, end, mode, direction)
    basesd = calc_basestd(x, start, end, mode, direction)

    def _calcz(y):
        return y \
            .divide(basesd) \
            .set({'system:time_start': y.get('system:time_start')})

    return anom.map(_calcz)







# 新增：计算稳定Z分数
def calc_robust_zscore(x, start, end, mode="IW", direction="DESCENDING"):
    """
    计算每幅图像的稳定Z分数 RZ = (X-median)/1.4826*MAD
    """
    scaling_factor = 1.4826  # 标准化常数

    # 1.计算中位数异常
    median_anom = calc_median_anomaly(x, start, end, mode, direction)

    # 2.计算基线MAD(已处理零值)
    basemad_safe = calc_basemad(x,start,end,mode,direction)

    # 3.计算分母
    denominator = basemad_safe.multiply(scaling_factor)

    # 4.定义内部函数计算稳健Z分数
    def _calcrz(y_anom):
        robust_z = y_anom.divide(denominator)
        return robust_z.set({'system:time_start': y_anom.get('system:time_start')})

    # 5.应用到中位数异常集合上
    return median_anom.map(_calcrz)


