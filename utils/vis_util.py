import jax
import jax.numpy as jnp
import numpy as np

def make_grid_visualization(vis, grid=4, max_bz=4, to_uint8=False, is_pt=False):
  if to_uint8:
    vis = float_to_uint8(vis)

  if is_pt:
    vis = np.transpose(vis, (0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)

  assert vis.ndim == 4, vis.shape
  n, h, w, c = vis.shape

  col = grid
  row = min(grid, n // col) 
  if n % (col * row) != 0:
    n = col * row * max_bz
    vis = vis[:n]
    n, h, w, c = vis.shape
  assert n % (col * row) == 0

  vis = vis.reshape((-1, col, row * h, w, c))
  vis = jnp.einsum('mlhwc->mhlwc', vis)
  vis = vis.reshape((-1, row * h, col * w, c))

  bz = min(vis.shape[0], max_bz)
  vis = vis[:bz][0]
  return jax.device_get(vis)

def float_to_uint8(vis):
  if not isinstance(vis, np.ndarray):
    vis = jax.device_get(vis)
  # -1, 1 -> 0, 255
  vis = vis * 0.5 + 0.5 # 0-1
  vis = vis * 255.0
  vis = np.clip(vis, 0, 255)
  vis = vis.astype(np.uint8)
  return vis

VIS_PROMPTS = [
  # 无尽的书之迷宫，地下城市中的书架地牢与发光蘑菇，梦书之城风格
  'endless book labyrinth illustration for The City of Dreaming Books by walter moers bookshelf dungeon glowing mushrooms scenic light under city underground rustic old place ar 169', 

  # 丛林中的美洲豹，覆盖huichol串珠装饰，炫彩幻觉色彩，电影感光影
  'beautiful Jaguar decorated with huichol beads, in the jungle, plants everywhere, DMT colours, ultra realistic , cinematic lighting   v 5', 

  # 水果与花朵形成的河流，绽放般爆炸的意象
  'river of fruits and flowers exploding ', 

  # 夜雨中1950年代老爷车，背景有树木，35mm胶片风格
  'rain pouring down on a vintage car from 1950 era and trees are in the background of the car. Its a night shot 35 mm', 

  # 天蓝色的奶茶纸杯，上面覆盖鲜奶油
  'a baby blue paper cup for bubble tea with whipped cream on top', 

  # 夜晚豪华顶楼公寓窗边的非洲女王形象，色彩绚烂、杂志级摄影风
  'gorgeous african queen gal standing by the window supercool swanky penthouse interior at night, amazing lighting, architectural digest photograph, brilliant colors, chaos, anarchy, liberty, independence, soul and afropop vibes, very detailed, photo taken with Hasselblad X1D, ISO 100, national geographic ', 

  # 真菌植物学插画，现实主义风格的裸盖菇
  'botanical drawing, psylocibe cubensis, realistic', 

  # 说唱歌手Notorious B.I.G坐在金色王座上，地狱风红色背景，超写实
  'Notorious B.I.G sitting slumped over a golden throne, with a kings crown, wearing a fur coat, at the back of the throne, a red color, as if he was in hell, color coded, ultradetailed, ultrarealistic, ultra high quality, ultra high definition, Careful composition, sharpen, insane details, cinematic lights, photorealism, 30mm shot Shutter Speed 1125, F5.6, White Balance, Megapixel, Pro Photo RGB, Unreal Engine, Cinematic, Chromatic Aberration, 8k, 4k ', 

  # 未来科幻城市中心，霓虹灯与飞行载具，Blade Runner风格
  'Futuristic scifi city center, bustling crowds, tier 2 civilization, advanced technology, towering skyscrapers, neon lights, flying vehicles, inspired by the art of Syd Mead and the movie Blade Runner, vibrant and dynamic urban landscape ', 

  # 亚马逊雨林河流风景
  'Amazon Rainforest river ', 

  # 抽象画：灯光下用餐，亚洲女孩面对发光餐盘
  'Abtract painting, Dinner with lights is an abstract picture. Asian girl sitting ready to eat a glowing plate ', 

  # 多种室内绿植，颜色造型各异的花盆摆放在架子上
  'assortment of house plants in pots of various shapes and colors, placed on shelves ', 

  # 微观童话精灵王国，实验室微距摄影风格
  'fantastical microscopic fairy kingdom photographed by science lab, microscopic art, 8k, UHD ', 

  # 年轻非裔美国消防员，全身形象，背光黑暗背景，超写实
  'a young African American fireman in worn out fireman gear, backlit against a dark background. Full body depiction. Hyper realistic, highly detailed, 8k, photo.', 

  # 西班牙乡村中渐进式文艺复兴风堡垒城市，电影级光影与细节
  'progressive renaissance medieval fortress city in spanish country side. Cinematic, Color Grading, Photography, Shot on 50mm lense, Ultra  Wide Angle, intricate details, beautifully color graded, Unreal Engine, Cinematic, Editorial Photography, Shot on 85mm lens, White Balance, Halfrear Lighting, Backlight, Natural Lighting, Cinematic Lighting, Studio Lighting, Global Illumination, Screen Space Global Illumination, Ray Tracing Global Illumination, Optics, Scattering, Ray Tracing Reflections, Lumen Reflections, Screen Space Reflections, Chromatic Aberration, Ray Traced, Ray Tracing Ambient Occlusion, Anti  Aliasing, FKAA, TXAA, RTX, SSAO, Shaders, OpenGL  Shaders, GLSL  Shaders, Post Processing, Post  Production, Tone Mapping, CGI, 4k, high detailed, ', 

  # 海滩狂欢派对上，身着rave装束的克林特·伊斯特伍德操作DJ设备的电影剧照
  'a High Definition, cinematic movie still of Clint Eastwood using complicated DJ equipment while dressed like a raver on the beach at an all night rave party'
]