"""트라이브 빌리지 게임 그림: 무료 SDXL Turbo 계열 모델(적은 단계라 CPU에서도 빠름) → 배경 지우기(rembg)
python tools/gen_tribez.py <id> <seed>   결과: tribez/out/<id>_s<seed>.png (투명 배경)"""
import json, sys, time, os, torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

item_id, seed = sys.argv[1], int(sys.argv[2])
cfg = json.load(open("tribez/prompts.json", encoding="utf-8"))
it = next(i for i in cfg["items"] if i["id"] == item_id)
MODEL = os.environ.get("MODEL", "Lykon/dreamshaper-xl-v2-turbo")
torch.set_num_threads(os.cpu_count())
t0 = time.time()
pipe = AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=torch.bfloat16, use_safetensors=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
print(f"모델 로드 {time.time()-t0:.0f}초", flush=True)
steps = int(os.environ.get("STEPS", "8"))
def cb(p, i, t, kw):
    print(f"step {i+1}/{steps} {time.time()-t0:.0f}s", flush=True); return kw
kind = it.get("kind", "b")
style = cfg["_style"][kind] if isinstance(cfg["_style"], dict) else cfg["_style"]
neg = cfg["_negative"] + (", " + cfg.get("_neg_extra", {}).get(kind, "") if cfg.get("_neg_extra", {}).get(kind) else "")
img = pipe(prompt=it["prompt"] + ", " + style, negative_prompt=neg,
           width=it["w"], height=it["h"], num_inference_steps=steps, guidance_scale=2.0,
           generator=torch.Generator().manual_seed(seed), callback_on_step_end=cb).images[0]
os.makedirs("tribez/out", exist_ok=True)
img.convert("RGB").save(f"tribez/out/{item_id}_s{seed}_raw.jpg", quality=85)
from rembg import remove, new_session
cut = remove(img, session=new_session("isnet-general-use"))
bb = cut.getchannel("A").getbbox()
if bb:
    cut = cut.crop(bb)
cut.save(f"tribez/out/{item_id}_s{seed}.png")
# 게임 크기 (가로 최대 512, 인물은 높이 256)
g = cut.copy()
lim = (512, 512) if kind == "b" else ((256, 256) if kind == "c" else (256, 256))
g.thumbnail(lim)
os.makedirs("tribez/game", exist_ok=True)
g.save(f"tribez/game/{item_id}_s{seed}.png")
print(f"완료 {time.time()-t0:.0f}초")
