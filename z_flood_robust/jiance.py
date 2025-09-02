!pip install geemap
%pip install pygis

import ee
import geemap

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize(project='geemap-441216')

从谷歌云盘里加载z-flood文件夹

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/Colab_Notebooks/')

try:
    import z_flood
    # 尝试调用 z_flood 中的函数
    # 这里假设你已经在 __init__.py 中导入了 calc_basemean
    # 如果没有，你需要使用 z_flood.zscore.calc_basemean
    z_flood.calc_basemean
    print("z_flood 包已成功导入，并且可以访问其中的函数")
except AttributeError:
    print("z_flood 包可能已导入，但无法访问其中的函数/变量")
except ImportError:
     print("导入z_flood失败")

# 1.导入依赖

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle #用于在图表中添加形状(例如矩形)
import re #正则表达式模块
from z_flood import calc_basemean,calc_basestd,calc_zscore
from z_flood import mapFloods,floodPalette

from ipywidgets import Label #用于创建交互式控件

# 2.定义交互式地图以选择感兴趣区域

#解析用户点击地图时生成的坐标字符串，提取经纬度并返回
def parseClickedCoordinates(label):
  #利用正则表达式，从label.value提取出坐标：一个可能带有负号的浮点数 这里为【经度，纬度】
  #正则表达式： r'(?:-)?[0-9]+.[0-9]+'
  #r'表示原始字符，(?:-)是非捕获组，?:表示负号是可选的，[0-9]+表示匹配一个数字，+表示可以出现一次或多次，.匹配小数点，最后再次匹配一个或多个数字
  coords = [float(c) for c in re.findall(r'(?:-)?[0-9]+.[0-9]+', label.value)]
  coords.reverse() #反转为【纬度，经度】，符合GEE坐标格式
  return coords

#创建一个Lable，用于显示用户点击的目标
l = Label()
display(1)

#处理用户与地图的交互事件，当用户点击地图时，将点击的坐标存储到Label控件中
def handle_interaction(**kwargs):
  #kwargs包含交互事件的参数 kwargs.get('type'):获取事件类型为鼠标点击 kwargs.get('coordinates')：获取点击的坐标，转换为字符串并存储到Label控件中
  if kwargs.get('type') == 'click':
    l.value = str(kwargs.get('coordinates'))

#创建交互式地图
Map = geemap.Map()
Map.on_interaction(handle_interaction)
Map

# 3.定义几何范围并展示

lon,lat = parseClickedCoordinates(l)
w,h = 0.1,0.1 #矩形宽度与高度（单位：度）

geometry = ee.Geometry.Polygon(
    [[[lon-w,lat-h],
     [lon-w,lat+h],
     [lon+w,lat+h],
     [lon+w,lat-h]]]
)

#将几何范围添加到地图
Map.addLayer(
    geometry,
    {'color':'red','fillColor':'00000000'},
    'AOI'
)
Map

## 附：通过geemap的功能：绘制多边形

roi = Map.user_roi

if roi is not None:
  #获取ROI类型
  roi_type = roi.type().getInfo()
  print(f"ROI 类型：{roi_type}")

  #如果是Polygon，获取坐标
  if roi_type == 'Polygon':
    coords = roi.coordinates().getInfo()
    print(f"ROI 坐标：{coords}")

    geometry_roi = ee.Geometry.Polygon(coords)
    Map.addLayer(
        geometry_roi,
        {'color':'yellow','fillColor':'00000000'},
        'AOI2'
    )
Map


# 4.过滤Sentinel-1数据

targdate = '2017-05-16' #目标日期（想要监测洪水的日期)
basestart = '2016-12-30' #基线的开始和结束日期
baseend = '2017-04-15'

aoi2 = ee.Geometry.Rectangle([-90.4920,38.9835,-90.14153,38.81116])

filters = [
    ee.Filter.listContains("transmitterReceiverPolarisation", "VV"),
    #ee.Filter.listContains("transmitterReceiverPolarisation", "VH"),
    ee.Filter.equals("instrumentMode", "IW"),
    ee.Filter.equals("orbitProperties_pass","ASCENDING"),
    ee.Filter.date('2015-01-01', ee.Date(targdate).advance(1, 'day'))
]
#时间范围：从2015年1月1日到目标日期后一天


#加载S1数据并计算Z分数
s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filter(filters).filterBounds(aoi2)
z = calc_zscore(s1,basestart,baseend,'IW','ASCENDING')
#时间范围：从2015年1月1日到目标日期后一天




# 定义一个函数，将日期信息添加到每个图像
def addDate(image):
  date = image.date().millis() #获取日期毫秒表示
  return image.addBands(ee.Image.constant(date).rename('date').long())
#将函数应用到图像集和中每个图像
s1_with_date = s1.map(addDate)
mosaic_image = s1_with_date.mosaic().clip(aoi2)
Map.addLayer(mosaic_image.select('date'), {'min': ee.Date('2015-01-01').millis(), 'max': ee.Date(targdate).millis()}, 'Date (milliseconds)')
Map

# 5.显示Z分数图层

Map = geemap.Map()
Map.setCenter(lon,lat,11)
#{0}表示占位符，表示在字符串该位置中插入一个变量(将目标日期格式化)
Map.addLayer(s1.mosaic().clip(aoi2).select('VV'), {'min': -25, 'max': 0}, 'VV Backscatter (dB); {0}'.format(targdate))

#z分数的颜色
zpalette = ['#b2182b','#ef8a62','#fddbc7','#f7f7f7','#d1e5f0','#67a9cf','#2166ac']
clipped_z = z.map(lambda image: image.clip(geometry))
Map.addLayer(clipped_z.mosaic().select('VV'), {'min': -5, 'max': 5, 'palette': zpalette}, 'VV Z-score; {0}'.format(targdate))

#用于点击地图某一点，生成Z分数时间序列图的点击函数(点击地图传递坐标)
label = Label()
def handle_interaction(**kwargs):
  if kwargs.get('type') == 'click':
    coords = kwargs.get('coordinates')
    label.value = str(coords)

Map.on_interaction(handle_interaction)
Map


image_to_display = s1.mosaic()
image_to_display.getInfo()

# 6.提取时间序列数据

#定义一个提取p所在位置的S-1后向散射数据和Z分数数据的时间序列，并将这些数据整理成一个Pandas DataFrame
def get_ts(p):
  #按时间升序排列，提取p所在位置及周围30m范围内所有影像的像元值
  x = s1.filter(ee.Filter.equals('instrumentMode', 'IW')).sort('system:time_start').getRegion(p,scale=30).getInfo()
  xz = z.getRegion(p,scale=30).getInfo()
  #x 与 xz的结构比较特殊：所有的元素都是列表，第一个元素是表头（x[0])，其余的元素，每一个都包含了某个时间点影像的信息
  #例：x[1],x[2],x[3]..每一行都是不同的Sentinel-1影像在p这个位置各种的属性与像元值
  #例：x[1][0]是第一个影像的ID,x[2][4]是第二个影像的VV波段值
  #作切片操作，得到表头以外的数据
  x = x[1:]
  xz = xz[1:]
  #创建所有S1原影像和所有Z分数影像的dataframe,用字典指定DataFrame的每一列
  s1df = pd.DataFrame({
      'ID':[y[0] for y in x],
      'VV':[y[4] for y in x],
      'VH':[y[5] for y in x],
  })

  zdf = pd.DataFrame({
    'ID': [y[0] for y in xz],
    'ZVV': [y[4] for y in xz],
    'ZVH': [y[5] for y in xz]
  })
  #日期解析函数
  def get_date(f):
    #使用正则表达式查找匹配'数字+T+数字'的模式，例如：20141003T120344，即影像名称，取出第一个匹配项
    datestr = re.findall(r'[0-9]+T[0-9]+',f)[0]
    #将字符串解析为日期时间格式
    return datetime.strptime(datestr,"%Y%m%dT%H%M%S")
  #为s1和z的两个dataframe添加日期列
  s1df = s1df.assign(date = [get_date(i) for i in s1df['ID']])
  zdf = zdf.assign(date = [get_date(i) for i in zdf['ID']])
  #合并两个dataframe
  #使用'date'作为连接键，采用内连接，即只保留两个date都相同的行，最后选择最终dataframe要保留的列
  df = s1df.merge(zdf,'inner',on = 'date')[['date', 'VV', 'VH', 'ZVV', 'ZVH']]
  return df

coords = parseClickedCoordinates(label)
print('取样点的坐标是：',coords)
p = ee.Geometry.Point(coords)
df = get_ts(p).query("date > '2015-01-01'")
df




# 7.可视化时间序列

#用plotly,创建图表和子图
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#创建子图
fig = make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=("Backscatter","Z-score"))

#第一个子图(VV,VH后向散射)
#绘制VV,VH
fig.add_trace(go.Scatter(x=df['date'],y=df['VV'],mode='markers',name='VV',marker = dict(symbol='circle',color='blue')),row=1,col=1) #只显示点的散点图
fig.add_trace(go.Scatter(x=df['date'],y=df['VH'],mode='markers',name='VH',marker = dict(symbol='circle',color='red')),row=1,col=1)

#第二个子图(Z分数)
#绘制ZVV和ZVH
fig.add_trace(go.Scatter(x=df['date'],y=df['ZVV'],mode='markers',name='ZVV',marker = dict(symbol='circle',color='blue')),row=2,col=1)
fig.add_trace(go.Scatter(x=df['date'],y=df['ZVH'],mode='markers',name='ZVH',marker = dict(symbol='circle',color='red')),row=2,col=1)
#添加Z分数阈值线
#z=0 灰色实线：
fig.add_trace(go.Scatter(x=[df['date'].min(),df['date'].max()],y=[0,0],mode='lines',name='Z=0',line=dict(color='gray',dash='solid')),row=2,col=1)
#z=-2 粉红色虚线
fig.add_trace(go.Scatter(x=[df['date'].min(),df['date'].max()],y=[-2,-2],mode='lines',name='Z=-2',line=dict(color='deeppink',dash='dash')),row=2,col=1)

#添加基线期阴影
if isinstance(basestart,str):
  basestart = pd.to_datetime(basestart)
if isinstance(baseend,str):
  baseend = pd.to_datetime(baseend)
fig.add_shape(type='rect',x0=basestart,x1=baseend,y0=df[['VV','VH']].min().min(),y1=df[['VV','VH']].max().max(),fillcolor='lightblue',opacity=0.2,line_width=0,layer='below',row=1,col=1)
fig.add_shape(type="rect",x0=basestart, x1=baseend, y0=df[['ZVV', 'ZVH']].min().min(), y1=df[['ZVV', 'ZVH']].max().max(),fillcolor="lightblue", opacity=0.2, line_width=0,layer="below", row=2, col=1)

# --- 添加垂直日期线 (目标日期) ---
if isinstance(targdate, str):
   targdate = pd.to_datetime(targdate)

fig.add_vline(x=targdate, line_color="gray", line_dash="dot",row='all',col=1)

# --- 更新布局 ---

fig.update_layout(
    title=f"Sentinel-1 Time Series at Lon: {coords[0]:.2f}, Lat: {coords[1]:.2f}",
    xaxis_title="Date",
    yaxis_title="Backscatter Coefficient (dB)",
    yaxis2_title="Z-score",
    legend_title="Legend",
    hovermode="x unified",
    height=800
)
# 调整图例位置（可选）
fig.update_layout(legend=dict(
    orientation="h",  # 水平放置
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig.show()


# 8.洪水制图

from os import POSIX_SPAWN_CLOSE
zvv_thd = -2.5
zvh_thd = -2.5
pin_thd = 10 #历史水体概率阈值
pow_thd = 90 #永久水体概率阈值

floods = mapFloods(clipped_z.mosaic(),zvv_thd,zvh_thd,pin_thd=pin_thd,pow_thd=pow_thd,use_dswe=False)
kernel = ee.Kernel.circle(radius=1)
opened_floods = floods.focal_min(kernel=kernel,iterations=1).focal_max(kernel=kernel,iterations=1)
opened_floods = opened_floods.updateMask(opened_floods.neq(0))

#形态学处理


Map.addLayer(opened_floods,{'min':0,'max':20,'palette':floodPalette},"Flood Map_1_9类,{0}".format(targdate))
Map.setCenter(coords[0],coords[1],10)
Map


#重分类洪水并可视化
reclassified_floods = opened_floods.remap([1,2,3,10,11,12,13,20],[1,1,1,2,2,2,2,3])

#定义新的颜色与标签
colors = ['#FF0000', '#800080', '#00008B']
labels = ['New Flooding','Prior Inundation','Permanent Water']
#可视化
Map.addLayer(reclassified_floods,{'min':1,'max':3,'palette':colors},'Flood Map_2_3类')
Map.add_legend(title='Flood Classification', labels=labels, colors=colors)
Map

8.1 加入DSWE的洪水制图

import sys
from google.colab import drive
drive.mount('/content/drive')
sys.path.append('/content/drive/MyDrive/Colab_Notebooks/')
from eedswe import dswe,cdswe

#初步洪水分类合并：除了0和20的类别都合并
preliminary_floods = floods.remap([1,2,3,10,11,12,13,20],[1,1,1,1,1,1,1,0])

#计算DSWE概率（2005-2018）并生成图
dswe_filters = [
    ee.Filter.date('2005-01-01','2018-12-31'),
    ee.Filter.calendarRange(1,12,'month'),
]
dswe_probs = cdswe(bounds=geometry,filters=dswe_filters)
dswe_total_prob = dswe_probs.select('pDSWE1').add(dswe_probs.select('pDSWE2')).add(dswe_probs.select('pDSWE3'))


#洪水淹没范围分级
flood_severity = ee.Image(0)  # 创建一个初始影像，所有像素值为 0
flood_severity = flood_severity.where(preliminary_floods.eq(1).And(dswe_total_prob.gte(0).And(dswe_total_prob.lte(10))), 1)  # 最严重
flood_severity = flood_severity.where(preliminary_floods.eq(1).And(dswe_total_prob.gte(20).And(dswe_total_prob.lte(50))), 2)  # 中等
flood_severity = flood_severity.where(preliminary_floods.eq(1).And(dswe_total_prob.gte(50).And(dswe_total_prob.lte(100))), 3)  # 次等
flood_severity = flood_severity.where(floods.eq(20),4) #永久水体

#可视化：
colors = ['#FF0000', '#FFA500', '#FFFF00', '#00008B']
labels = ['Severe Flooding', 'Moderate Flooding', 'Minor Flooding','Permanent Water']
# --- 添加分级后的洪水图层 ---
Map.addLayer(flood_severity.updateMask(flood_severity.neq(0)), {'min': 1, 'max': 4, 'palette': colors}, f"Flood Severity ({targdate})")

# --- 添加图例 ---
Map.add_legend(title='Flood Severity', labels=labels, colors=colors)

# --- 设置地图中心 ---
Map.setCenter(coords[0], coords[1], 10)

# --- 显示地图 ---
Map

#计算各严重度洪水淹没面积
levels = [1,2,3,4]
level_names = ['Severe Flooding', 'Moderate Flooding', 'Minor Flooding','Permanent Water']  # 对应等级的名称

Pixelarea = ee.Image.pixelArea() #创建像元面积影像 每个像素的值代表该像素的实际面积

for level,name in zip(levels,level_names):
  #创建二值淹没
  mask = flood_severity.eq(level)
  #计算面积
  area_image = Pixelarea.updateMask(mask) #只计算严重性1，2，3，4洪水的区域
  area_stats = area_image.reduceRegion(
      reducer=ee.Reducer.sum(),
      geometry=geometry,
      scale=10,
      maxPixels=1e13,
      bestEffort=True
  )
  #获取面积(平方米)
  area_sqm = ee.Number(area_stats.get('area'))
  # 转换为平方公里 (可选)
  area_sqkm = area_sqm.divide(1e6)

  # 打印结果
  print(f"{name} Area: {area_sqkm.getInfo():.2f} sq. km")

