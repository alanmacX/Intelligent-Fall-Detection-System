import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance


def process_and_save_black_text(video_path, output_dir="patent_output_black"):
    print(f"\n🎬 [开始] 处理视频: {video_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("❌ 视频无法读取，请检查路径")
        return

    indices = np.linspace(0, total_frames - 1, 8, dtype=int)
    frames_pil = []

    print(f"ℹ️ 采样帧索引: {indices}")

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame.fill(255)
        else:
            frame = cv2.resize(frame, (224, 224))


        cv2.putText(frame, str(i + 1), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6)

        cv2.putText(frame, str(i + 1), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(frame_rgb))

    cap.release()

    if len(frames_pil) != 8:
        print("❌ 帧数不足")
        return

    def make_grid(img_list):
        grid = Image.new('RGB', (448, 448))
        grid.paste(img_list[0], (0, 0))
        grid.paste(img_list[1], (224, 0))
        grid.paste(img_list[2], (0, 224))
        grid.paste(img_list[3], (224, 224))
        return grid

    phase1_color = make_grid(frames_pil[0:4])
    phase2_color = make_grid(frames_pil[4:8])

    def to_patent_style(pil_img):
        gray = pil_img.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        high_contrast = enhancer.enhance(1.3)
        return high_contrast

    phase1_bw = to_patent_style(phase1_color)
    phase2_bw = to_patent_style(phase2_color)

    combined_bw = Image.new('L', (896, 448))
    combined_bw.paste(phase1_bw, (0, 0))
    combined_bw.paste(phase2_bw, (448, 0))

    abs_out_dir = os.path.abspath(output_dir)
    name = os.path.basename(video_path).split('.')[0]

    p1_path = os.path.join(abs_out_dir, f"{name}_P1_BW.jpg")
    p2_path = os.path.join(abs_out_dir, f"{name}_P2_BW.jpg")
    comb_path = os.path.join(abs_out_dir, f"{name}_Combined_BlackNum.jpg")

    phase1_bw.save(p1_path)
    phase2_bw.save(p2_path)
    combined_bw.save(comb_path)

    print("\n" + "=" * 40)
    print("✅✅✅ 黑白专利附图（黑色序号版）已生成！")
    print(f"预览图: {comb_path}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    target_video = "/home/alanmac/fall/fall_dataset/Falls/Figshare_Fall_ACT4_R_1_20240917123626_17634736569720.mp4"

    process_and_save_black_text(target_video)