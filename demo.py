import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import logging
import time
import yaml
from PIL import Image
from torchvision import transforms
from dotmap import DotMap

# ==============================================================================
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(CURRENT_DIR, "lib")
FASTVLM_DIR = os.path.join(LIB_DIR, "FastVLM")
LLAVA_REPO_ROOT = "weights/llava-fastvithd_1.5b_stage3/llava-fastvithd_1.5b_stage3"

paths_to_add = [LIB_DIR, FASTVLM_DIR, LLAVA_REPO_ROOT]
for p in paths_to_add:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from ActionCLIP.clip import clip
    from ActionCLIP.modules.Visual_Prompt import visual_prompt
except ImportError:
    logging.error("❌ 严重错误: 无法导入 ActionCLIP")
    sys.exit(1)

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
except ImportError:
    logging.warning("⚠️ 警告: 无法导入 FastVLM")

# ==============================================================================
# ==============================================================================
CLASSES = [
    "A video of a person bending down with control to pick up something.",
    "A video of a person lying comfortably on a bed, sofa, or floor to rest or read.",
    "A video of a person performing normal daily activities safely.",
    "A video of a person intentionally sitting down on a chair or sofa.",
    "A video of a person standing up or standing still safely.",
    "A video of a person walking normally and steadily in a room.",
    "A video of a person suddenly losing consciousness and collapsing to the ground.",
    "A video of a person losing balance uncontrollably and crashing down.",
    "A video of a person lying motionless on the ground after a dangerous accident.",
    "A video of a person slowly sliding down against a wall or object unable to stand.",
    "A video of a person struggling painfully on the floor unable to get up.",
    "A video of a person falling down quickly and hitting the floor violently."
]

CLASS_LABELS = [
    "ADL - Bending", "ADL - Lying/Rest", "ADL - Safe Activity",
    "ADL - Sitting", "ADL - Standing", "ADL - Walking",
    "FALL - Collapse", "FALL - Loss Balance", "FALL - Motionless",
    "FALL - Slow Slide", "FALL - Struggle", "FALL - Violent"
]

FALL_IDXS = [6, 7, 8, 9, 10, 11]


class LiteRouter(nn.Module):
    def __init__(self, input_dim=514, hidden_dim=256):
        super(LiteRouter, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 0
            nn.ReLU(),  # 1
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),  # 3
            nn.ReLU(),  # 4
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 7
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# ==============================================================================
class GuardianCognition:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 [系统] 初始化 Guardian Engine (Device: {self.device})")

        self.config_path = os.path.join(CURRENT_DIR, "configs/custom.yaml")
        self.ac_weights = os.path.join(CURRENT_DIR, "weights/model_best.pt")
        self.router_path = os.path.join(CURRENT_DIR, "weights/router_best.pth")
        self.vlm_path = os.path.join(CURRENT_DIR, "weights/llava-fastvithd_1.5b_stage3/llava-fastvithd_1.5b_stage3")

        print("   ├── 1. 加载 ActionCLIP...")
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = DotMap(yaml.safe_load(f))
        else:
            self.config = DotMap({"network": {"arch": "ViT-B/16", "sim_header": "Transf"}, "data": {"num_segments": 8}})

        clip_source = self.ac_weights if os.path.exists(self.ac_weights) else self.config.network.arch
        self.clip_model, clip_state_dict = clip.load(clip_source, device=self.device, jit=False)
        self.clip_model.eval()
        self.fusion_model = visual_prompt(
            self.config.network.sim_header, clip_state_dict=clip_state_dict, T=self.config.data.num_segments
        ).to(self.device)
        self.fusion_model.eval()

        if os.path.exists(self.ac_weights):
            checkpoint = torch.load(self.ac_weights, map_location=self.device)

            def rm_pfx(d):
                return {k.replace('module.', ''): v for k, v in d.items()}

            sd = rm_pfx(checkpoint['model_state_dict']) if 'model_state_dict' in checkpoint else rm_pfx(checkpoint)
            fusion_sd = checkpoint.get('fusion_model_state_dict') if isinstance(checkpoint, dict) else None
            self.clip_model.load_state_dict(sd, strict=False)
            if fusion_sd is not None:
                self.fusion_model.load_state_dict(rm_pfx(fusion_sd), strict=False)
        else:
            print("   ❌ ActionCLIP权重缺失")

        with torch.no_grad():
            text_inputs = clip.tokenize(CLASSES).to(self.device)
            self.text_features = self.clip_model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.transform = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        print("   ├── 2. 加载 Router...")
        self.router = LiteRouter(input_dim=514).to(self.device)
        self.router.eval()
        if os.path.exists(self.router_path):
            try:
                self.router.load_state_dict(torch.load(self.router_path, map_location=self.device))
                print("   │   ✅ 已加载")
            except Exception as e:
                print(f"   │   ❌ 加载失败: {e}")
                self.router = None
        else:
            print("   │   ⚠️ 文件不存在")
            self.router = None

        print("   └── 3. 加载 FastVLM...")
        self.vlm_model = None
        if os.path.exists(self.vlm_path):
            try:
                logging.getLogger("transformers").setLevel(logging.ERROR)
                model_name = get_model_name_from_path(self.vlm_path)
                self.tokenizer, self.vlm_model, self.image_processor, _ = load_pretrained_model(
                    model_path=self.vlm_path, model_base=None, model_name=model_name,
                    load_8bit=False, load_4bit=False, device=self.device
                )
                print("       ✅ 就绪")
            except Exception as e:
                print(f"       ❌ 失败: {e}")
        else:
            print("       ⚠️ 路径无效")

    def infer_actionclip(self, frames, return_features=False):
        if self.fusion_model is None: return (None, None) if return_features else None

        imgs = [self.transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames]
        input_tensor = torch.stack(imgs).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        with torch.no_grad():
            b, c, t, h, w = input_tensor.size()
            image_input = input_tensor.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)

            image_features = self.clip_model.encode_image(image_input).view(b, t, -1)
            image_features = torch.nan_to_num(image_features, nan=0.0)
            video_features = self.fusion_model(image_features)
            video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-8)

            probs = (100.0 * video_features @ self.text_features.T).softmax(dim=-1).float().cpu().numpy()[0]

        if return_features:
            probs_safe = np.clip(probs, 1e-7, 1.0)
            probs_safe = probs_safe / np.sum(probs_safe)
            entropy = -np.sum(probs_safe * np.log(probs_safe))
            sorted_probs = np.sort(probs_safe)[::-1]
            margin = sorted_probs[0] - sorted_probs[1]

            meta_features = torch.tensor([[entropy, margin]], device=self.device, dtype=torch.float32)
            combined_features = torch.cat([video_features.float(), meta_features], dim=1)

            return probs, combined_features

        return probs

    def bayesian_route(self, feats, samples=20):
        if self.router is None: return 1.0, 0.0, 0.0

        def enable_dropout(m):
            if type(m) == nn.Dropout:
                m.train()

        self.router.apply(enable_dropout)

        batch_feats = feats.repeat(samples, 1)  # [T, 514]
        with torch.no_grad():
            mc_outputs = self.router(batch_feats).squeeze()  # [T]

        self.router.eval()

        mean_pred = mc_outputs.mean().item()
        uncertainty = mc_outputs.std().item()

        risk_sensitivity = 3.0
        final_score = mean_pred + (risk_sensitivity * uncertainty)
        final_score = min(max(final_score, 0.0), 1.0)

        return final_score, uncertainty, mean_pred

    def infer_fastvlm(self, frames):
        if self.vlm_model is None: return "ERROR", "VLM Not Loaded"

        annotated_frames = []
        for i, frame in enumerate(frames):
            img_draw = frame.copy()
            cv2.putText(img_draw, str(i + 1), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
            annotated_frames.append(Image.fromarray(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)))

        def stitch_2x2(img_list):
            if not img_list: return Image.new('RGB', (224, 224))
            w, h = img_list[0].size
            grid = Image.new('RGB', (w * 2, h * 2))
            for i, img in enumerate(img_list):
                grid.paste(img, ((i % 2) * w, (i // 2) * h))
            return grid

        current_frames = annotated_frames
        while len(current_frames) < 8:
            current_frames.append(current_frames[-1])

        image_phase1 = stitch_2x2(current_frames[:4])
        image_phase2 = stitch_2x2(current_frames[4:8])

        qs = (
            "You are provided with two images representing a continuous video sequence.\n"
            "Image 1 (Frames 1-4): The beginning phase.\n"
            "Image 2 (Frames 5-8): The ending phase.\n"
            "Compare the posture in Image 1 versus Image 2. "
            "Did the person transition from standing/sitting to lying on the ground? "
            "Answer strictly with 'CONCLUSION: FALL' or 'CONCLUSION: SAFE'."
        )

        if self.vlm_model.config.mm_use_im_start_end:
            qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" +
                  DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs)
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(self.device)
        image_tensor = process_images([image_phase1, image_phase2], self.image_processor, self.vlm_model.config)

        with torch.inference_mode():
            output_ids = self.vlm_model.generate(
                input_ids,
                images=image_tensor.half().to(self.device),
                image_sizes=[image_phase1.size, image_phase2.size],
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                use_cache=True
            )

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().upper()

        if "CONCLUSION: FALL" in output:
            return "FALL", output
        elif "CONCLUSION: SAFE" in output:
            return "SAFE", output
        if "FALL" in output and "NOT FALL" not in output: return "FALL", output
        return "SAFE", output


def load_frames(video_path, num_frames=8):
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: return None
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames


def main():
    VIDEO_PATH = "/home/alanmac/fall/ActionCLIP/5.MP4"
    ROUTER_THRESH = 0.3
    # 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥

    print("\n" + "=" * 60)
    print("🚀 启动 Guardian 系统 (自适应贝叶斯路由 + 双图流)")
    print("=" * 60)
    engine = GuardianCognition()

    print(f"\n📂 读取视频: {VIDEO_PATH}")
    t_start = time.time()
    frames = load_frames(VIDEO_PATH)
    if not frames:
        print("❌ 无法读取视频")
        return
    print(f"✅ 完成 (耗时 {(time.time() - t_start) * 1000:.1f}ms)")

    # 1. ActionCLIP
    print("\n---------------- [1] ActionCLIP (快速感知) ----------------")
    t1 = time.time()
    probs, feats = engine.infer_actionclip(frames, return_features=True)
    t1_cost = (time.time() - t1) * 1000

    top1_idx = probs.argmax()
    ac_result = "🔴 FALL" if top1_idx in FALL_IDXS else "🟢 SAFE"
    print(f"   预测: {ac_result} | {CLASS_LABELS[top1_idx]}")
    print(f"   置信度: {probs[top1_idx]:.4f}")
    print(f"   耗时: {t1_cost:.1f}ms")

    # 2. Router
    print("\n---------------- [2] Router (贝叶斯决策) ----------------")
    t2 = time.time()
    should_route = False

    if engine.router:
        final_score, uncertainty, raw_pred = engine.bayesian_route(feats, samples=20)
        should_route = final_score > ROUTER_THRESH

        print(f"   基础预测: {raw_pred:.4f}")
        print(f"   不确定性: {uncertainty:.4f}")
        print(f"   最终得分: {final_score:.4f} (阈值 {ROUTER_THRESH})")

        decision_msg = "🚨 拦截 (复核)" if should_route else "✅ 放行 (资源节省)"
        print(f"   最终决策: {decision_msg}")
    else:
        print("   ⚠️ Router 未激活")

    t2_cost = (time.time() - t2) * 1000

    # 3. FastVLM
    print("\n---------------- [3] FastVLM (双图流时序分析) ----------------")
    final_verdict = ac_result
    source = "ActionCLIP"
    t3_cost = 0.0

    if should_route:
        print("   呼叫大模型...")
        t3 = time.time()
        vlm_res, vlm_txt = engine.infer_fastvlm(frames)
        t3_cost = (time.time() - t3) * 1000
        print(f"   VLM输出: [{vlm_txt}] -> {vlm_res}")
        final_verdict = "🔴 FALL" if vlm_res == "FALL" else "🟢 SAFE"
        source = "FastVLM (Override)"
    else:
        print("   (无需介入)")

    print("\n" + "=" * 60)
    print(f"🔍 最终判定: {final_verdict} (来源: {source})")
    print(f"⏱️ 总耗时:   {t1_cost + t2_cost + t3_cost:.1f}ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
