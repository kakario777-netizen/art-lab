"""무료 SDXL(애니메이션 일러스트 모델)로 CPU에서 그림 한 장 생성
- init_search: 위키미디어 공용 사진을 밑그림으로 (img2img)
- ref: 참고 인물 그림으로 같은 인물 유지 (IP-Adapter)
"""
import json, sys, time, os, gc, urllib.request, urllib.parse, io
import torch
from PIL import Image, ImageOps
from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,
                       EulerAncestralDiscreteScheduler)

item_id, seed = sys.argv[1], int(sys.argv[2])
cfg = json.load(open("prompts.json", encoding="utf-8"))
it = next(i for i in cfg["items"] if i["id"] == item_id)
MODEL = "cagliostrolab/animagine-xl-4.0"
torch.set_num_threads(os.cpu_count())
flags = open("/proc/cpuinfo").read()
# bf16 하드웨어 지원 CPU는 bf16이 빠르고, 없는 CPU는 fp32가 훨씬 빠름
DT = torch.bfloat16 if ("avx512_bf16" in flags or "amx_bf16" in flags) else torch.float32
print("정밀도:", DT, flush=True)
os.makedirs("out", exist_ok=True)
t0 = time.time()
W, H = it["w"], it["h"]


def commons_photo(query):
    """위키미디어 공용에서 검색해 첫 사진을 받음"""
    ua = {"User-Agent": "art-lab-test/1.0 (github actions; game background test)"}
    q = urllib.parse.urlencode({"action": "query", "generator": "search", "gsrsearch": "filetype:bitmap " + query,
                                "gsrnamespace": 6, "gsrlimit": 10, "prop": "imageinfo", "iiprop": "url|size",
                                "iiurlwidth": 1200, "format": "json"})
    for attempt in range(4):
        try:
            data = json.load(urllib.request.urlopen(urllib.request.Request(
                "https://commons.wikimedia.org/w/api.php?" + q, headers=ua), timeout=60))
            pages = sorted(data["query"]["pages"].values(), key=lambda p: p.get("index", 99))
            for p in pages:
                ii = p.get("imageinfo", [{}])[0]
                url = ii.get("thumburl") or ii.get("url")
                if not url or ii.get("width", 0) < 800:
                    continue
                print("밑그림:", p["title"], url, flush=True)
                raw = urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=120).read()
                return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            print("위키미디어 재시도", attempt, e, flush=True)
            time.sleep(30 * (attempt + 1))
    return None


pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=DT, use_safetensors=True)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
print(f"모델 로드 {time.time()-t0:.0f}초", flush=True)

prompt = it["prompt"] + ", " + cfg["_style"]
steps = int(os.environ.get("STEPS", "28"))
kw = dict(negative_prompt=cfg["_negative"], guidance_scale=5.5,
          generator=torch.Generator().manual_seed(seed))

# 같은 인물: 참고 그림을 IP-Adapter로
if it.get("ref"):
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                         image_encoder_folder="models/image_encoder", torch_dtype=DT)
    pipe.set_ip_adapter_scale(it.get("ip_scale", 0.6))
    ref = Image.open(it["ref"]).convert("RGBA")
    bgw = Image.new("RGBA", ref.size, (255, 255, 255, 255)); bgw.alpha_composite(ref)
    ref = bgw.convert("RGB")
    embeds = pipe.prepare_ip_adapter_image_embeds(ip_adapter_image=ref, ip_adapter_image_embeds=None,
                                                  device="cpu", num_images_per_prompt=1,
                                                  do_classifier_free_guidance=True)
    pipe.image_encoder = None; gc.collect()  # 메모리 확보
    kw["ip_adapter_image_embeds"] = embeds
    print(f"참고 인물 준비 {time.time()-t0:.0f}초", flush=True)

init = commons_photo(it["init_search"]) if it.get("init_search") else None
if init is not None:
    init = ImageOps.fit(init, (W, H))
    init.save(f"out/{item_id}_s{seed}_밑그림.jpg", quality=85)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
    kw.update(image=init, strength=it.get("strength", 0.7))
else:
    kw.update(width=W, height=H)


def cb(p, i, t, k):
    print(f"step {i+1} {time.time()-t0:.0f}s", flush=True)
    return k


img = pipe(prompt=prompt, num_inference_steps=steps, callback_on_step_end=cb, **kw).images[0]
img.save(f"out/{item_id}_s{seed}.png")
print(f"완료 {time.time()-t0:.0f}초")
