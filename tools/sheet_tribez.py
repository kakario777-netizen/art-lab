"""tribez/out 결과를 분류(건물·자연·인물·초상)별 한 장씩 + 전체 한 장으로 모음"""
import glob, json, os, sys
from PIL import Image, ImageDraw
cfg = json.load(open("tribez/prompts.json", encoding="utf-8"))
cat = {i["id"]: i.get("cat", "기타") for i in cfg["items"]}
fs = sorted(f for f in glob.glob("tribez/out/*.png"))
if not fs: sys.exit()
def sheet(files, out, W=260):
    cols = min(6, len(files)); rows = (len(files) + cols - 1) // cols
    s = Image.new("RGB", (cols * W, rows * (W + 24)), (120, 170, 90)); d = ImageDraw.Draw(s)
    for k, f in enumerate(files):
        im = Image.open(f).convert("RGBA"); im.thumbnail((W - 10, W - 10))
        x, y = (k % cols) * W, (k // cols) * (W + 24)
        s.paste(im, (x + (W - im.width) // 2, y + 24 + (W - im.height) // 2), im)
        d.text((x + 6, y + 6), os.path.basename(f)[:-4], fill="black")
    s.save(out, quality=86)
groups = {}
for f in fs:
    iid = os.path.basename(f).rsplit("_s", 1)[0]
    groups.setdefault(cat.get(iid, "기타"), []).append(f)
names = {"건물": "buildings", "자연": "nature", "인물": "characters", "초상": "portraits"}
for g, files in groups.items():
    sheet(files, f"tribez/sheet_{names.get(g, 'etc')}.jpg")
sheet(fs, "tribez/sheet.jpg", W=200)
