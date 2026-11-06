"""tribez/out 의 투명 배경 그림을 체크무늬 위에 번호 붙여 한 장에"""
import glob, sys
from PIL import Image, ImageDraw
fs = sorted(f for f in glob.glob("tribez/out/*.png") if not f.endswith("_raw.png"))
if not fs: sys.exit()
W = 300
cols = min(4, len(fs)); rows = (len(fs) + cols - 1) // cols
s = Image.new("RGB", (cols * W, rows * (W + 26)), (120, 170, 90)); d = ImageDraw.Draw(s)
for k, f in enumerate(fs):
    im = Image.open(f).convert("RGBA"); im.thumbnail((W - 10, W - 10))
    x, y = (k % cols) * W, (k // cols) * (W + 26)
    s.paste(im, (x + (W - im.width) // 2, y + 26 + (W - im.height) // 2), im)
    d.text((x + 6, y + 6), f.split("/")[-1], fill="black")
s.save("tribez/sheet.jpg", quality=88)
