"""배경 지우기 + 반투명 정리: rembg 가 남긴 흐린 알파를 단단하게 만들고, 안쪽 구멍은 메움.
python tools/cut_tribez.py   → tribez/out/*_raw.jpg 전부 다시 오려서 out/·game/ 덮어씀"""
import glob, json, os
import numpy as np
from PIL import Image
from scipy import ndimage
from rembg import remove, new_session

_S = None
def cut(img):
    """rembg 마스크 + '테두리에서 이어진 배경색' 빼기를 합쳐 물체를 정하고, 안쪽은 완전 불투명, 가장자리 1~2px만 부드럽게."""
    global _S
    _S = _S or new_session("isnet-general-use")
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    m = np.asarray(remove(img, session=_S, only_mask=True), dtype=np.float32) / 255
    # 배경: 테두리 색과 비슷하고 테두리에 이어진 영역
    bd = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    ref = np.median(bd, 0)
    near = np.abs(rgb - ref).sum(-1) < 38
    lab, _ = ndimage.label(near)
    edge_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, edge_ids[edge_ids > 0])
    obj = (m > 0.25) | (~bg & (m > 0.02))          # rembg 가 놓친 부분도 배경색이 아니면 물체
    obj &= ~(bg & (m < 0.6))                          # 배경색인데 rembg 도 자신 없으면 배경
    obj = ndimage.binary_opening(obj, iterations=1)
    lab, n = ndimage.label(obj)
    if n:
        sz = ndimage.sum(obj, lab, range(1, n + 1))
        obj = np.isin(lab, [k + 1 for k, v in enumerate(sz) if v >= sz.max() * 0.03])
    obj = ndimage.binary_fill_holes(ndimage.binary_closing(obj, iterations=2))
    # 안쪽은 255, 바깥 2px 만 부드럽게
    d = ndimage.distance_transform_edt(obj)
    a = np.clip(d / 2.0, 0, 1)
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
