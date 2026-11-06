"""이번에 만든 그림만 이름 붙여 한 장에 모아 보기 (참고 인물 그림은 맨 앞에)"""
import glob, sys, os
from PIL import Image, ImageDraw
ids = [x.strip() for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else None
fs = sorted(f for f in glob.glob("out/*") if f.endswith((".png", ".jpg"))
            and (ids is None or any(os.path.basename(f).startswith(i + "_s") for i in ids)))
if ids and "hero_ip" in ids and os.path.exists("ref/hero.png"):
    fs = ["ref/hero.png"] + fs
if not fs: sys.exit()
W = 300
ims = []
for f in fs:
    i = Image.open(f).convert("RGBA"); b = Image.new("RGBA", i.size, "white"); b.alpha_composite(i)
    i = b.convert("RGB"); ims.append(i.resize((W, int(i.height * W / i.width))))
H = max(i.height for i in ims); cols = min(5, len(ims)); rows = (len(ims) + cols - 1) // cols
s = Image.new("RGB", (cols * W, rows * (H + 26)), "white"); d = ImageDraw.Draw(s)
for k, (f, i) in enumerate(zip(fs, ims)):
    x, y = (k % cols) * W, (k // cols) * (H + 26)
    s.paste(i, (x, y + 26)); d.text((x + 6, y + 7), f.replace("out/", ""), fill="black")
s.save("sheet.jpg", quality=88)
