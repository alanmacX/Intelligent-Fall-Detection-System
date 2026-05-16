import cv2
import torch
import torch.nn as nn
import sys
import os
import logging
import numpy as np
from PIL import Image
from torchvision import transforms
import yaml
from dotmap import DotMap

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
LIB_DIR = os.path.join(ROOT_DIR, "lib")
if LIB_DIR not in sys.path: sys.path.append(LIB_DIR)

FASTVLM_DIR = os.path.join(LIB_DIR, "FastVLM")
if FASTVLM_DIR not in sys.path: sys.path.append(FASTVLM_DIR)

try:
    from ActionCLIP.clip import clip
    from ActionCLIP.modules.Visual_Prompt import visual_prompt
except ImportError:
    logging.error("❌ ActionCLIP 模块缺失")

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
except ImportError:
    logging.warning("⚠️ FastVLM 模块缺失")

# ========================================================
# ========================================================

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


class RhythmRouter(nn.Module):
    def __init__(self, input_dim=515, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ========================================================

class GuardianCognition:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"🧠 [认知层] 初始化 (Device: {self.device})")

        # 1. ActionCLIP
        self.config_path = os.path.join(ROOT_DIR, "configs/custom.yaml")
        self.ac_weights = os.path.join(ROOT_DIR, "weights/model_best.pt")
        self.router_path = os.path.join(ROOT_DIR, "weights/router_rhythm_best.pth")

        logging.info("🏋️ [认知层] 加载 ActionCLIP...")

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.config = DotMap(config_dict)
        else:
            self.config = DotMap({"network": {"arch": "ViT-B/16", "sim_header": "Transf"}, "data": {"num_segments": 8}})

        clip_source = self.ac_weights if os.path.exists(self.ac_weights) else self.config.network.arch
        self.clip_model, clip_state_dict = clip.load(clip_source, device=self.device, jit=False)
        self.clip_model.eval()

        self.fusion_model = visual_prompt(
            self.config.network.sim_header,
            clip_state_dict=clip_state_dict,
            T=self.config.data.num_segments
        ).to(self.device)
        self.fusion_model.eval()

        if os.path.exists(self.ac_weights):
            checkpoint = torch.load(self.ac_weights, map_location=self.device)

            def rm_pfx(d):
                return {k.replace('module.', ''): v for k, v in d.items()}

            state_dict = rm_pfx(checkpoint['model_state_dict']) if 'model_state_dict' in checkpoint else rm_pfx(checkpoint)
            fusion_state_dict = checkpoint.get('fusion_model_state_dict') if isinstance(checkpoint, dict) else None

            self.clip_model.load_state_dict(state_dict, strict=False)
            if fusion_state_dict is not None:
                self.fusion_model.load_state_dict(rm_pfx(fusion_state_dict), strict=False)
            logging.info("✅ ActionCLIP 权重加载成功")
        else:
            logging.error(f"❌ 权重缺失: {self.ac_weights}")

        with torch.no_grad():
            text_inputs = clip.tokenize(CLASSES).to(self.device)
            self.text_features = self.clip_model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.transform = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        # 2. Bayesian Router with rhythm surprise
        self.router = None
        if os.path.exists(self.router_path):
            try:
                router_ckpt = torch.load(self.router_path, map_location=self.device)
                router_cfg = router_ckpt.get("config", {})
                self.router = RhythmRouter(
                    input_dim=router_cfg.get("input_dim", 515),
                    hidden_dim=router_cfg.get("hidden_dim", 256),
                ).to(self.device)
                self.router.load_state_dict(router_ckpt["model_state_dict"])
                self.router.eval()
                logging.info("✅ Rhythm Bayesian Router 权重加载成功")
            except Exception as e:
                logging.error(f"❌ Rhythm Bayesian Router 加载失败: {e}")

        # 3. FastVLM
        self.vlm_path = os.path.join(ROOT_DIR, "weights/llava-fastvithd_1.5b_stage3/llava-fastvithd_1.5b_stage3")
        self.vlm_model = None

        if os.path.exists(self.vlm_path):
            logging.info("🔮 [认知层] 加载 FastVLM...")
            try:
                model_name = get_model_name_from_path(self.vlm_path)
                self.tokenizer, self.vlm_model, self.image_processor, _ = load_pretrained_model(
                    model_path=self.vlm_path, model_base=None, model_name=model_name,
                    load_8bit=False, load_4bit=False, device=self.device
                )
                self.vlm_query = "请分析画面。第一步回答'FALL'或'SAFE'。第二步用中文描述人物姿态。"
                logging.info("✅ FastVLM 就绪")
            except Exception as e:
                logging.error(f"❌ FastVLM 加载失败: {e}")

    def infer_actionclip(self, frame_buffer, return_features=False):
        if self.fusion_model is None: return None

        imgs = [self.transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frame_buffer]
        input_tensor = torch.stack(imgs).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        with torch.no_grad():
            b, c, t, h, w = input_tensor.size()
            image_input = input_tensor.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)

            image_features = self.clip_model.encode_image(image_input).view(b, t, -1)

            video_features = self.fusion_model(image_features)
            video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-8)

            probs = (100.0 * video_features @ self.text_features.T).softmax(dim=-1).float().cpu().numpy()[0]

        if return_features:
            probs_safe = np.clip(probs, 1e-7, 1.0)
            probs_safe = probs_safe / probs_safe.sum()
            entropy = float(-np.sum(probs_safe * np.log(probs_safe)))
            sorted_probs = np.sort(probs_safe)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1])
            meta = torch.tensor([[entropy, margin]], device=self.device, dtype=torch.float32)
            feats = torch.cat([video_features.float(), meta], dim=1)
            return probs, feats, {"entropy": entropy, "margin": margin}

        return probs

    def bayesian_route(self, feats_514, rhythm_surprise, samples=5):
        if self.router is None:
            return 1.0, 0.0, 1.0

        rhythm_tensor = torch.tensor([[rhythm_surprise]], device=self.device, dtype=torch.float32)
        router_input = torch.cat([feats_514.float(), rhythm_tensor], dim=1)

        def enable_dropout(module):
            if isinstance(module, nn.Dropout):
                module.train()

        self.router.apply(enable_dropout)
        batch = router_input.repeat(samples, 1)
        with torch.no_grad():
            outputs = self.router(batch).view(-1)
        self.router.eval()
        mean_score = float(outputs.mean().item())
        uncertainty = float(outputs.std(unbiased=False).item())
        final_score = float(np.clip(mean_score + 1.5 * uncertainty, 0.0, 1.0))
        return final_score, uncertainty, mean_score

    def infer_fastvlm(self, frame_buffer, infrared=False):
        if self.vlm_model is None:
            return "ERROR", "VLM Not Loaded"
        if not isinstance(frame_buffer, (list, tuple)):
            frame_buffer = [frame_buffer]

        annotated_frames = []
        for i, frame in enumerate(frame_buffer[:8]):
            img_draw = frame.copy()
            cv2.putText(img_draw, str(i + 1), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
            annotated_frames.append(Image.fromarray(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)))
        while len(annotated_frames) < 8 and annotated_frames:
            annotated_frames.append(annotated_frames[-1])
        if not annotated_frames:
            return "ERROR", "No frames"

        def stitch_2x2(images):
            w, h = images[0].size
            grid = Image.new("RGB", (w * 2, h * 2))
            for idx, img in enumerate(images[:4]):
                grid.paste(img, ((idx % 2) * w, (idx // 2) * h))
            return grid

        image_phase1 = stitch_2x2(annotated_frames[:4])
        image_phase2 = stitch_2x2(annotated_frames[4:8])
        env_prompt = ""
        if infrared:
            env_prompt = (
                "Current input is Infrared Night Vision. Ignore texture loss. "
                "Focus strictly on skeletal geometry and posture changes.\n"
            )
        qs = (
            DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n"
            + env_prompt
            + "You are provided with two images representing frames 1-4 and 5-8 of a video. "
            + "Compare the person's posture and location over time. "
            + "Answer with 'CONCLUSION: FALL' or 'CONCLUSION: SAFE', then one short Chinese reason."
        )

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        image_tensor = process_images([image_phase1, image_phase2], self.image_processor, self.vlm_model.config)

        with torch.inference_mode():
            output_ids = self.vlm_model.generate(
                input_ids,
                images=image_tensor.half().to(self.device),
                image_sizes=[image_phase1.size, image_phase2.size],
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
            )
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        upper = output.upper()
        if "CONCLUSION: FALL" in upper or ("FALL" in upper and "NOT FALL" not in upper):
            return "FALL", output
        return "SAFE", output
