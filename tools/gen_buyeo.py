"""부여전기 그림: 무료 SDXL Turbo 계열 모델 → 배경 지우기(rembg) → 128px 도트풍 가공
python tools/gen_buyeo.py <id> <seed>
결과: buyeo/out/<id>_s<seed>.png (배경만 지운 원본), buyeo/game/<id>_s<seed>.png (128px 가공본)"""
import json, sys, time, os, torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

item_id, seed = sys.argv[1], int(sys.argv[2])
cfg = json.load(open("buyeo/prompts.json", encoding="utf-8"))
it = next(i for i in cfg["items"] if i["id"] == item_id)
MODEL = os.environ.get("MODEL", "Lykon/dreamshaper-xl-v2-turbo")
torch.set_num_threads(os.cpu_count())
t0 = time.time()
pipe = AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=torch.bfloat16, use_safetensors=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
steps = int(os.environ.get("STEPS", "8"))
kind = it.get("kind", "i")
img = pipe(prompt=it["prompt"] + ", " + cfg["_style"][kind], negative_prompt=cfg["_negative"],
           width=it["w"], height=it["h"], num_inference_steps=steps, guidance_scale=2.0,
           generator=torch.Generator().manual_seed(seed)).images[0]
os.makedirs("buyeo/out", exist_ok=True); os.makedirs("buyeo/game", exist_ok=True)
img.convert("RGB").save(f"buyeo/out/{item_id}_s{seed}_raw.jpg", quality=85)
if kind == "x":   # 이어 붙는 질감: 가장자리 섞어 이음새 없애기
    im = img.convert("RGB").resize((512, 512), Image.LANCZOS); a = np.asarray(im).astype(float)
    r = np.roll(np.roll(a, 256, 0), 256, 1); yy, xx = np.mgrid[0:512, 0:512]
    w = (np.minimum(np.minimum(xx, 511 - xx), np.minimum(yy, 511 - yy)) / 256.0).clip(0, 1)[..., None]
    out = a * w + r * (1 - w)
    Image.fromarray(out.astype(np.uint8)).save(f"buyeo/game/{item_id}_s{seed}.png"); im.save(f"buyeo/out/{item_id}_s{seed}.png")
    print(f"완료 {time.time()-t0:.0f}초"); sys.exit(0)
if kind == "e":   # 빛 효과: 검은 배경 → 투명
    a = np.asarray(img.convert("RGB").resize((256, 256), Image.LANCZOS)).astype(float)
    al = a.max(2); out = np.dstack([a / np.maximum(al, 1)[..., None] * 255, al]).clip(0, 255).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(f"buyeo/game/{item_id}_s{seed}.png"); Image.fromarray(out, "RGBA").save(f"buyeo/out/{item_id}_s{seed}.png")
    print(f"완료 {time.time()-t0:.0f}초"); sys.exit(0)
if kind == "u" and "hanji" in item_id:
    im = img.convert("RGB").resize((512, 512), Image.LANCZOS); im.save(f"buyeo/game/{item_id}_s{seed}.png"); im.save(f"buyeo/out/{item_id}_s{seed}.png"); sys.exit(0)
if kind in ("t", "m"):   # 큰 그림: 배경 그대로, 크기만
    tw = 1280 if kind == "t" else 768
    im = img.convert("RGB").resize((tw, round(img.height * tw / img.width)), Image.LANCZOS)
    im.save(f"buyeo/game/{item_id}_s{seed}.png"); im.save(f"buyeo/out/{item_id}_s{seed}.png")
    print(f"완료 {time.time()-t0:.0f}초"); sys.exit(0)
from rembg import remove, new_session
cut = remove(img, session=new_session("isnet-general-use"))
bb = cut.getchannel("A").getbbox()
if bb: cut = cut.crop(bb)
cut.save(f"buyeo/out/{item_id}_s{seed}.png")

if "fit" in it:   # 지도 물체: 예전 그림 크기(2배) 칸에 아래 가운데 맞춤
    FW, FH = it["fit"]
    c = cut.copy(); c.thumbnail((FW - 4, FH - 2), Image.LANCZOS)
    a = np.asarray(c).copy(); mask = a[..., 3] >= 128
    rgb = Image.fromarray(a[..., :3]).quantize(colors=40, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
    a[..., :3] = np.asarray(rgb); a[..., 3] = np.where(mask, 255, 0)
    canvas = np.zeros((FH, FW, 4), np.uint8)
    oy, ox = FH - a.shape[0], (FW - a.shape[1]) // 2
    canvas[oy:, ox:ox + a.shape[1]] = a
    m = canvas[..., 3] > 0; ring = np.zeros_like(m)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)): ring |= np.roll(np.roll(m, dy, 0), dx, 1)
    ring &= ~m; canvas[ring] = (34, 22, 14, 255)
    Image.fromarray(canvas, "RGBA").save(f"buyeo/game/{item_id}_s{seed}.png")
    print(f"완료 {time.time()-t0:.0f}초"); sys.exit(0)

# ── 도트풍 가공: 112px 안에 맞춤 → 색 24개 → 알파 딱 자르기 → 1px 짙은 외곽선 → 128px 칸 가운데
S, INNER, NCOL = (192, 184, 32) if kind == "p" else ((384, 376, 64) if kind == "u" else (128, 112, 24))   # 초상은 크게·색 많이
c = cut.copy(); c.thumbnail((INNER, INNER), Image.LANCZOS)
a = np.asarray(c).copy()
mask = a[..., 3] >= 128
rgb = Image.fromarray(a[..., :3]).quantize(colors=NCOL, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
a[..., :3] = np.asarray(rgb); a[..., 3] = np.where(mask, 255, 0)
canvas = np.zeros((S, S, 4), np.uint8)
oy, ox = (S - a.shape[0]) // 2, (S - a.shape[1]) // 2
canvas[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a
m = canvas[..., 3] > 0
ring = np.zeros_like(m)
for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    ring |= np.roll(np.roll(m, dy, 0), dx, 1)
ring &= ~m
canvas[ring] = (34, 22, 14, 255)
Image.fromarray(canvas, "RGBA").save(f"buyeo/game/{item_id}_s{seed}.png")
print(f"완료 {time.time()-t0:.0f}초")
