import cv2
import torch
import sys
import os
import logging
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


# ========================================================

class GuardianCognition:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"🧠 [认知层] 初始化 (Device: {self.device})")

        # 1. ActionCLIP
        self.config_path = os.path.join(ROOT_DIR, "configs/custom.yaml")
        self.ac_weights = os.path.join(ROOT_DIR, "weights/model_best.pt")

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

        # 2. FastVLM
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

    def infer_actionclip(self, frame_buffer):
        if self.fusion_model is None: return None

        imgs = [self.transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frame_buffer]
        input_tensor = torch.stack(imgs).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        with torch.no_grad():
            b, c, t, h, w = input_tensor.size()
            image_input = input_tensor.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)

            image_features = self.clip_model.encode_image(image_input).view(b, t, -1)

            video_features = self.fusion_model(image_features)
            video_features /= video_features.norm(dim=-1, keepdim=True)

            probs = (100.0 * video_features @ self.text_features.T).softmax(dim=-1).float().cpu().numpy()[0]

        return probs

    def infer_fastvlm(self, frame_buffer):
        return "FALL", "检测到跌倒"
