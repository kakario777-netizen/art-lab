"""엠버폴 3D 시험: AI 그림 1장 → TripoSR 로 3D 모델 → 간단한 걷기·공격 동작(정점 변형) → 8방향 스프라이트
python tools/ember3d.py <id>   (emberfall/3dtest/<id>/ 에 결과)"""
import json, os, sys, math, subprocess, time
import numpy as np
from PIL import Image
iid = sys.argv[1]
OUT = f"emberfall/3dtest/{iid}"; os.makedirs(OUT, exist_ok=True)
cfg = json.load(open("emberfall/prompts.json", encoding="utf-8"))
it = next(i for i in cfg["items"] if i["id"] == iid)
t0 = time.time()
# 1) 정면 A자세 그림
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
torch.set_num_threads(os.cpu_count())
pipe = AutoPipelineForText2Image.from_pretrained(os.environ.get("MODEL", "Lykon/dreamshaper-xl-v2-turbo"), torch_dtype=torch.bfloat16, use_safetensors=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
prompt = it["prompt"] + ", full body, front view, symmetrical standing A-pose, arms held slightly away from the body, legs apart, whole figure visible, dark fantasy game character, hand-painted texture, even lighting, plain white background"
img = pipe(prompt=prompt, negative_prompt="cropped, cut off, multiple, text, side view, back view, sitting, blurry, background scenery",
           width=1024, height=1024, num_inference_steps=8, guidance_scale=2.0, generator=torch.Generator().manual_seed(int(os.environ.get("SEED", "1")))).images[0]
img.save(f"{OUT}/concept.png"); del pipe
print(f"그림 {time.time()-t0:.0f}s", flush=True)
# 2) TripoSR → 메쉬 (정점 색)
subprocess.run([sys.executable, "TripoSR/run.py", f"{OUT}/concept.png", "--output-dir", f"{OUT}/tsr", "--device", "cpu", "--mc-resolution", "192"], check=True)
import trimesh
mesh = trimesh.load(f"{OUT}/tsr/0/mesh.obj", process=False)
if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
V = np.asarray(mesh.vertices, dtype=np.float32); F = np.asarray(mesh.faces, dtype=np.int32)
C = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
print(f"메쉬 {len(V)} 정점 {len(F)} 면, {time.time()-t0:.0f}s", flush=True)
# 축 정하기: 가장 긴 축 = 위, 나머지 중 넓은 축 = 좌우, 남은 축 = 앞뒤
ext = V.max(0) - V.min(0); up = int(np.argmax(ext)); rest = [a for a in range(3) if a != up]
lat = rest[0] if ext[rest[0]] >= ext[rest[1]] else rest[1]; fwd = [a for a in rest if a != lat][0]
V = V - (V.max(0) + V.min(0)) / 2
if np.mean(V[:, up][V[:, up] > 0]) < 0: pass
H = ext[up]; h = (V[:, up] - V[:, up].min()) / H          # 0 = 발, 1 = 머리
P = np.stack([V[:, lat], V[:, up], V[:, fwd]], 1)        # x 좌우, y 위, z 앞뒤
P[:, 1] -= P[:, 1].min(); P /= H                          # 키 1
LATW = np.abs(P[:, 0]).max()
def rot_x(pts, pivot_y, pivot_z, ang):
    y = pts[:, 1] - pivot_y; z = pts[:, 2] - pivot_z; c, s = math.cos(ang), math.sin(ang)
    out = pts.copy(); out[:, 1] = pivot_y + y * c - z * s; out[:, 2] = pivot_z + y * s + z * c; return out
def pose(kind, ph):
    Q = P.copy()
    if kind == "walk":
        sw = math.sin(ph * 2 * math.pi) * .45
        leg = h < .46; side = np.sign(P[:, 0])
        for sg in (-1, 1):
            m = leg & (side == sg); w = np.clip((.46 - h[m]) / .46, 0, 1) ** .5
            Q[m] = rot_x(P[m], .46, 0, 0) ; ang = sg * sw
            q = rot_x(P[m], .46, 0, ang); Q[m] = P[m] + (q - P[m]) * w[:, None]
        arm = (h > .45) & (h < .82) & (np.abs(P[:, 0]) > LATW * .55)
        for sg in (-1, 1):
            m = arm & (side == sg); q = rot_x(P[m], .78, 0, -sg * sw * .8); Q[m] = q
        Q[:, 1] += abs(math.sin(ph * 2 * math.pi)) * .02
    elif kind == "attack":
        a = math.sin(min(1, ph * 1.6) * math.pi)
        up_ = h > .45; Q[up_] = rot_x(P[up_], .45, 0, .45 * a)
        arm = (h > .45) & (np.abs(P[:, 0]) > LATW * .55); Q[arm] = rot_x(Q[arm], .8, 0, -1.6 * a)
    return Q
# 3) 렌더 (moderngl, EGL 헤드리스): 디아2처럼 비스듬히 내려다보는 정사영
import moderngl
ctx = moderngl.create_standalone_context(backend="egl")
prog = ctx.program(vertex_shader="""#version 330
in vec3 p; in vec3 c; in vec3 n; uniform mat4 M; out vec3 vc; out vec3 vn;
void main(){ gl_Position = M * vec4(p,1.0); vc = c; vn = mat3(M) * n; }""",
fragment_shader="""#version 330
in vec3 vc; in vec3 vn; out vec4 o;
void main(){ vec3 L = normalize(vec3(-0.5, 0.7, 0.6)); float d = max(dot(normalize(vn), L), 0.0);
 o = vec4(vc * (0.45 + 0.75 * d), 1.0); }""")
S = 256
fbo = ctx.framebuffer(color_attachments=[ctx.texture((S, S), 4)], depth_attachment=ctx.depth_renderbuffer((S, S)))
def normals(Q):
    fn = np.cross(Q[F[:, 1]] - Q[F[:, 0]], Q[F[:, 2]] - Q[F[:, 0]]); vn = np.zeros_like(Q)
    for k in range(3): np.add.at(vn, F[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)
def render(Q, az):
    a = math.radians(az); el = math.radians(35)
    Ry = np.array([[math.cos(a), 0, math.sin(a), 0], [0, 1, 0, 0], [-math.sin(a), 0, math.cos(a), 0], [0, 0, 0, 1]], np.float32)
    Rx = np.array([[1, 0, 0, 0], [0, math.cos(el), -math.sin(el), 0], [0, math.sin(el), math.cos(el), 0], [0, 0, 0, 1]], np.float32)
    T = np.eye(4, dtype=np.float32); T[1, 3] = -.5
    Sc = np.diag([1.5, 1.5, -.8, 1]).astype(np.float32)
    M = Sc @ Rx @ Ry @ T
    data = np.hstack([Q, C, normals(Q)]).astype("f4")[F.reshape(-1)]
    vbo = ctx.buffer(data.tobytes()); vao = ctx.vertex_array(prog, [(vbo, "3f 3f 3f", "p", "c", "n")])
    prog["M"].write(M.T.copy().tobytes())
    fbo.use(); ctx.enable(moderngl.DEPTH_TEST); ctx.clear(0, 0, 0, 0); vao.render()
    im = Image.frombytes("RGBA", (S, S), fbo.color_attachments[0].read()).transpose(Image.FLIP_TOP_BOTTOM)
    vao.release(); vbo.release(); return im
dirs = 8; frames = 8
for kind in ["walk", "attack"]:
    sheet = Image.new("RGBA", (S * frames, S * dirs))
    gif = []
    for d in range(dirs):
        for f in range(frames):
            im = render(pose(kind, f / frames), d * 45)
            sheet.paste(im, (f * S, d * S))
            if d in (1, 3):
                bg = Image.new("RGBA", (S, S), (38, 34, 30, 255)); bg.alpha_composite(im); gif.append(bg.convert("P"))
    sheet.save(f"{OUT}/{kind}_sheet.png")
    gif[0].save(f"{OUT}/{kind}.gif", save_all=True, append_images=gif[1:], duration=90, loop=0)
# 한눈에: 그림 + 8방향 첫 프레임
prev = Image.new("RGBA", (S * 5, S * 2), (38, 34, 30, 255))
c = Image.open(f"{OUT}/concept.png").resize((S * 2, S * 2)); prev.paste(c, (0, 0))
for d in range(dirs):
    prev.alpha_composite(render(pose("walk", .25), d * 45), (S * 2 + (d % 3) * S, (d // 3) * S) if d < 6 else (S * 2 + (d - 6) * S, S))
prev.convert("RGB").save(f"{OUT}/preview.jpg", quality=88)
print(f"완료 {time.time()-t0:.0f}s")
