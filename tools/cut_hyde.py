"""한양 숨바꼭질: 인물 그림 배경 지우기 (isnet-anime) → hyde/cut/<이름>.png"""
import json, os
from PIL import Image, ImageFilter
from rembg import remove, new_session
sess = new_session("isnet-anime")
os.makedirs("hyde/cut", exist_ok=True)
for src, name in json.load(open("hyde/cut_list.json", encoding="utf-8")):
    im = Image.open(src).convert("RGB")
    out = remove(im, session=sess, post_process_mask=True)
    a = out.getchannel("A").point(lambda v: 0 if v < 40 else (255 if v > 200 else v))
    a = a.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    out.putalpha(a)
    bb = a.getbbox()
    if bb:
        out = out.crop(bb)
    W, H = 1024, 1536  # 기존 인물 그림과 같은 틀, 아래 가운데 정렬
    k = min((W - 40) / out.width, (H - 30) / out.height)
    out = out.resize((int(out.width * k), int(out.height * k)), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(out, ((W - out.width) // 2, H - out.height - 10), out)
    canvas.save(f"hyde/cut/{name}.png", optimize=True)
    print("ok", name)
