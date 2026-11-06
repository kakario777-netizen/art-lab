"""3D 킷으로 찍은 걷기 그림을 AI로 약하게 덧칠 (img2img, 모든 장 같은 시드) → 깜빡임 시험
python tools/restyle_tribez.py <strength>   입력 tribez/restyle/in/*.png → 출력 tribez/restyle/out_<strength>/"""
import glob, os, sys, time, torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image, DPMSolverMultistepScheduler
st = float(sys.argv[1])
MODEL = os.environ.get("MODEL", "Lykon/dreamshaper-xl-v2-turbo")
torch.set_num_threads(os.cpu_count())
pipe = AutoPipelineForImage2Image.from_pretrained(MODEL, torch_dtype=torch.bfloat16, use_safetensors=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
prompt = ("cute chibi tribal villager, stylized 3d game character, soft hand-painted textures, painterly shading, "
          "warm sunny lighting, vibrant colors, casual mobile game art, plain white background")
neg = "blurry, lowres, deformed, extra limbs, text, watermark, photo, dark"
out = f"tribez/restyle/out_{st}"; os.makedirs(out, exist_ok=True)
from rembg import remove, new_session
sess = new_session("isnet-general-use")
frames = []
for f in sorted(glob.glob("tribez/restyle/in/*.png")):
    t0 = time.time()
    src = Image.open(f).convert("RGBA")
    big = src.resize((src.width * 5, src.height * 5), Image.LANCZOS)
    bg = Image.new("RGBA", big.size, "white"); bg.alpha_composite(big)
    img = pipe(prompt=prompt, negative_prompt=neg, image=bg.convert("RGB"), strength=st,
               num_inference_steps=max(8, int(round(4 / st))), guidance_scale=2.0,
               generator=torch.Generator().manual_seed(7)).images[0]
    cut = remove(img, session=sess)
    a = big.getchannel("A")                      # 원래 윤곽을 살짝 넓혀 써서 실루엣 유지
    from PIL import ImageFilter
    cut.putalpha(Image.composite(cut.getchannel("A"), a.filter(ImageFilter.MaxFilter(9)), a.filter(ImageFilter.MaxFilter(9))))
    small = cut.resize(src.size, Image.LANCZOS)
    small.save(f"{out}/{os.path.basename(f)}")
    frames.append((src, small))
    print(os.path.basename(f), f"{time.time()-t0:.0f}s", flush=True)
# 비교 GIF: 왼쪽 원래, 오른쪽 덧칠 (3배 확대)
gif = []
for s, d in frames:
    c = Image.new("RGBA", (s.width * 6, s.height * 3), (120, 170, 90, 255))
    c.alpha_composite(s.resize((s.width * 3, s.height * 3)), (0, 0))
    c.alpha_composite(d.resize((s.width * 3, s.height * 3)), (s.width * 3, 0))
    gif.append(c.convert("P", palette=Image.ADAPTIVE))
gif[0].save(f"{out}/compare.gif", save_all=True, append_images=gif[1:], duration=110, loop=0)
