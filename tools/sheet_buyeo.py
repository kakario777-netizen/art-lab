"""buyeo 결과 모음: 그림마다 [배경 지운 원본 | 128px 가공본(2배 확대)] 을 나란히, 게임 가방 색 바탕"""
import glob, json, os
from PIL import Image, ImageDraw, ImageFont
cfg = json.load(open("buyeo/prompts.json", encoding="utf-8"))
names = {i["id"]: i["name"] for i in cfg["items"]}
fs = sorted(glob.glob("buyeo/game/*.png"))
if fs:
    W, cols = 300, 4
    rows = (len(fs) + cols - 1) // cols
    s = Image.new("RGB", (cols * W * 2, rows * (W + 28)), (46, 34, 22)); d = ImageDraw.Draw(s)
    font = None
    for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
        try: font = ImageFont.truetype(fp, 18); break
        except Exception: pass
    for k, f in enumerate(fs):
        base = os.path.basename(f)[:-4]; iid = base.rsplit("_s", 1)[0]
        x, y = (k % cols) * W * 2, (k // cols) * (W + 28)
        raw = f"buyeo/out/{base}.png"
        if os.path.exists(raw):
            im = Image.open(raw).convert("RGBA"); im.thumbnail((W - 20, W - 20))
            s.paste(im, (x + (W - im.width) // 2, y + 28 + (W - im.height) // 2), im)
        g = Image.open(f).convert("RGBA").resize((256, 256), Image.NEAREST)
        s.paste(g, (x + W + (W - 256) // 2, y + 28 + (W - 256) // 2), g)
        d.text((x + 8, y + 8), f"{names.get(iid, iid)} ({base})  원본 | 가공", fill=(240, 220, 170), font=font)
    s.save("buyeo/sheet.jpg", quality=88)
