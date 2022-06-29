import os
from io import BytesIO
import streamlit as st
checkpoint_path = os.path.join(os.path.dirname(__file__), "weights")
pic_type = st.selectbox("选择上传方式：", ("上传图片", "从摄像头截取"))
if pic_type == "上传图片":
    pic = st.file_uploader("请上传图片", type=("jpg", "jpeg", "png", "bmp", "tiff"))
else:
    pic = st.camera_input("请从摄像头截取图片")
checkpoint = st.selectbox("选择模型：", os.listdir(checkpoint_path))
disable_gpu = st.checkbox("禁用GPU")
start = st.button("开始转换")
if start:
    if not pic:
        st.error("请上传图片！")
    else:
        pic = BytesIO(pic.getvalue())
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor, to_pil_image

        from model import Generator

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"


        def load_image(image_bytes):
            img = Image.open(image_bytes).convert("RGB")
            return img


        def run():
            net = Generator()
            net.load_state_dict(torch.load(os.path.join(checkpoint_path, checkpoint), map_location="cpu"))
            net.to(device).eval()
            st.info(f"加载模型：{checkpoint}")

            image = load_image(pic)

            with torch.no_grad():
                image = to_tensor(image).unsqueeze(0) * 2 - 1
                out = net(image.to(device), False).cpu()
                out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
                out = to_pil_image(out)

            result = BytesIO()
            out.save(result, format="JPEG")
            st.info("图像已保存！")
            st.image(result)
            st.download_button("下载图片", result.read(), file_name="cartoon.jpeg", mime="image/jpeg")

        with st.spinner("正在生成中……"):
            run()
