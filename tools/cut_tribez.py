"""배경 지우기 + 반투명 정리: rembg 가 남긴 흐린 알파를 단단하게 만들고, 안쪽 구멍은 메움.
python tools/cut_tribez.py   → tribez/out/*_raw.jpg 전부 다시 오려서 out/·game/ 덮어씀"""
import glob, json, os
import numpy as np
from PIL import Image
from scipy import ndimage
from rembg import remove, new_session

_S = None
def cut(img):
    global _S
    _S = _S or new_session("isnet-general-use")
    m = np.asarray(remove(img, session=_S, only_mask=True), dtype=np.float32)
    # 배경과 색이 비슷하면(회색 바위 등) 마스크 전체가 흐리게 나옴 → 그림마다 기준을 맞춤
    hi = float(np.percentile(m[m > 15], 90)) if (m > 15).sum() > 100 else 255.0
    a = np.clip((m - 0.2 * hi) / (0.45 * hi), 0, 1)
    solid = a > 0.5
    lab, n = ndimage.label(solid)                          # 가장 큰 덩어리 + 큰 덩어리만 남김
    if n:
        sz = ndimage.sum(solid, lab, range(1, n + 1))
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sz) if s >= sz.max() * 0.05])
        filled = ndimage.binary_fill_holes(keep)           # 몸통 안 구멍 메움
        a = np.where(filled & ~keep, 1.0, np.where(ndimage.binary_dilation(keep, iterations=2), a, 0))
    out = img.convert("RGBA"); out.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    bb = out.getchannel("A").getbbox()
    return out.crop(bb) if bb else out

def save(out, item_id, seed, kind):
    out.save(f"tribez/out/{item_id}_s{seed}.png")
    g = out.copy(); g.thumbnail((512, 512) if kind == "b" else (256, 256))
    os.makedirs("tribez/game", exist_ok=True); g.save(f"tribez/game/{item_id}_s{seed}.png")

if __name__ == "__main__":
    kinds = {i["id"]: i.get("kind", "b") for i in json.load(open("tribez/prompts.json", encoding="utf-8"))["items"]}
    for p in sorted(glob.glob("tribez/out/*_raw.jpg")):
        base = os.path.basename(p)[:-8]; item_id, s = base.rsplit("_s", 1)
        save(cut(Image.open(p).convert("RGB")), item_id, s, kinds.get(item_id, "b")); print(base, flush=True)
