"""생성된 그림을 번호 붙여 한 장에 모아 보기"""
import glob, sys
from PIL import Image, ImageDraw
fs = sorted(glob.glob("out/*.png"))
if not fs: sys.exit()
W = 384; ims = [Image.open(f).convert("RGB") for f in fs]
ims = [i.resize((W, int(i.height * W / i.width))) for i in ims]
H = max(i.height for i in ims); cols = min(4, len(ims)); rows = (len(ims) + cols - 1) // cols
s = Image.new("RGB", (cols * W, rows * (H + 30)), "white"); d = ImageDraw.Draw(s)
for k, (f, i) in enumerate(zip(fs, ims)):
    x, y = (k % cols) * W, (k // cols) * (H + 30)
    s.paste(i, (x, y + 30)); d.text((x + 6, y + 8), f.split("/")[-1], fill="black")
s.save("sheet.jpg", quality=88)
