"""무료 SDXL(애니메이션 일러스트 모델)로 CPU에서 그림 한 장 생성"""
import json, sys, time, os, torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

item_id, seed = sys.argv[1], int(sys.argv[2])
cfg = json.load(open("prompts.json", encoding="utf-8"))
it = next(i for i in cfg["items"] if i["id"] == item_id)
MODEL = os.environ.get("MODEL", "cagliostrolab/animagine-xl-4.0")
torch.set_num_threads(os.cpu_count())
t0 = time.time()
try:
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, use_safetensors=True)
except Exception as e:
    print("diffusers 형식 실패, 단일 파일로:", e)
    from huggingface_hub import hf_hub_download, list_repo_files
    f = [x for x in list_repo_files(MODEL) if x.endswith(".safetensors")][0]
    pipe = StableDiffusionXLPipeline.from_single_file(hf_hub_download(MODEL, f), torch_dtype=torch.bfloat16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
print(f"모델 로드 {time.time()-t0:.0f}초", flush=True)
steps = int(os.environ.get("STEPS", "28"))
def cb(p, i, t, kw):
    print(f"step {i+1}/{steps} {time.time()-t0:.0f}s", flush=True); return kw
img = pipe(prompt=it["prompt"] + ", " + cfg["_style"], negative_prompt=cfg["_negative"],
           width=it["w"], height=it["h"], num_inference_steps=steps, guidance_scale=5.5,
           generator=torch.Generator().manual_seed(seed), callback_on_step_end=cb).images[0]
os.makedirs("out", exist_ok=True)
img.save(f"out/{item_id}_s{seed}.png")
print(f"완료 {time.time()-t0:.0f}초")
