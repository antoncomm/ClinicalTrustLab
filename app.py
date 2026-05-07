from pathlib import Path
from typing import Dict, Tuple, Optional

import gradio as gr
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from ctl.models import EfficientNet, HFClassificationModel, ResNet
from ctl.utils.attacks import PGDAttack, UniversalNoiseAdd, CWAttack, \
    UAPAttack, HopSkipJumpAttack, PGDAttack
from ctl.utils.grad_cam import get_gradcam
import torch
import matplotlib.pyplot as plt

try:
    import pydicom
except ImportError:
    pydicom = None


BASE_DIR = Path(__file__).resolve().parent
STANDARD_DICOM_DIR = BASE_DIR / "standard_dicoms"

MODEL_OPTIONS = {
    "efficientnet_b0_ct" : (EfficientNet ,"weights/baseline_epoch_7.pth"),
    "resnet50_ct" : (ResNet, None),
    "vit_small_ct": (HFClassificationModel, None),
}

ATTACK_OPTIONS = [
    "Without corruption",
    "Universal noise",
    "FGSM", 
    "PGD",
    "CW",
    "UAP",
    "HopSkipJump",
]

INTERPRETATION_OPTIONS = [
    "Without interpretation",
    "GradCAM",
]


def list_standard_dicoms() -> list[str]:
    if not STANDARD_DICOM_DIR.exists():
        return []

    files = []
    for ext in ("*.dcm", "*.dicom", "*.DCM"):
        files.extend(STANDARD_DICOM_DIR.glob(ext))

    return sorted([f.name for f in files])

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr)

    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val - min_val < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - min_val) / (max_val - min_val)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr

def dicom_to_pil(path: str | Path) -> Image.Image:
    if pydicom is None:
        raise RuntimeError(
            "pydicom is not installed. Please install it with: pip install pydicom"
        )

    ds = pydicom.dcmread(str(path))
    pixel_array = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    pixel_uint8 = normalize_to_uint8(pixel_array)
    image = Image.fromarray(pixel_uint8).convert("L")
    return image

def prepare_display_image(
    img: Image.Image,
    size: Tuple[int, int] = (900, 900),
) -> Image.Image:
    return ImageOps.contain(img.convert("RGB"), size)

def _transform_output(x):
    mean =torch.tensor([0.1217, 0.1217, 0.1217])
    std =torch.tensor([0.2299, 0.2299, 0.2299])
    mean = mean.view(1, -1, 1, 1).to(x.device)
    std = std.view(1, -1, 1, 1).to(x.device)
    return (x - mean) / std

def _transform_input(x):
    mean =torch.tensor([0.1217, 0.1217, 0.1217])
    std =torch.tensor([0.2299, 0.2299, 0.2299])
    mean = mean.view(1, -1, 1, 1).to(x.device)
    std = std.view(1, -1, 1, 1).to(x.device)
    return x * std + mean

def apply_attack(
    img_np: np.ndarray, 
    attack_name: str, 
    model: torch.nn.Module,
    device: str, 
) -> np.ndarray:
    model.eval()

    img = torch.from_numpy(img_np.copy().transpose((2, 0, 1))).unsqueeze(0).to(device)
    img = _transform_output(img)
    predict = model(img)

    if attack_name == "Without corruption":
        img_attacked = img.clone()
    elif attack_name == "FGSM":
        attack = PGDAttack(
            model=model,
            mean=(0.1217, 0.1217, 0.1217),
            std=(0.2299, 0.2299, 0.2299),
            height=img.shape[2],
            width=img.shape[3],
            max_iter=1,
            nb_classes=2,
            eps=30/255,
            eps_step=10/255,
        )
        img_attacked = attack.generate(img.detach().clone(), (predict > 0.5).long())[0]
    elif attack_name == "PGD":
        attack = PGDAttack(
            model=model,
            mean=(0.1217, 0.1217, 0.1217),
            std=(0.2299, 0.2299, 0.2299),
            height=img.shape[2],
            width=img.shape[3],
            max_iter=3,
            nb_classes=2,
            eps=30/255,
            eps_step=10/255,
        )
        img_attacked = attack.generate(img.detach().clone(), (predict > 0.5).long())[0]
    elif attack_name == "CW":
        attack = CWAttack(
            model=model,
            mean=(0.1217, 0.1217, 0.1217),
            std=(0.2299, 0.2299, 0.2299),
            height=img.shape[2],
            width=img.shape[3],
            num_classes=2, 
            c=1,
            kappa=0.5, 
            lr=1/255,
            max_iter=3,
        )
        img_attacked = attack.generate(img.detach().clone(), (predict > 0.5).long())[0]
    elif attack_name == "Universal noise":
        attack = UniversalNoiseAdd(
            model=model,
            mean=(0.1217, 0.1217, 0.1217),
            std=(0.2299, 0.2299, 0.2299),
            height=img.shape[2],
            width=img.shape[3],
        )
        img_attacked = attack.generate(img.detach().clone(), (predict > 0.5).long())[0]
    elif attack_name == "UAP":
        attack = UAPAttack(
            model=model,
            mean=(0.1217, 0.1217, 0.1217),
            std=(0.2299, 0.2299, 0.2299),
            height=img.shape[2],
            width=img.shape[3],
            eps=10/255,
            lr=1/255,
            max_iter=10,
            is_color=True,
        )
        img_attacked = attack.generate(img.detach().clone(), (predict > 0.5).long())[0]
    elif attack_name == "HopSkipJump":
        attack = HopSkipJumpAttack(
            model=model,
            mean=(0.1217, 0.1217, 0.1217),
            std=(0.2299, 0.2299, 0.2299),
            height=img.shape[2],
            width=img.shape[3],
            max_iter=10,
            init_trials=10,
            grad_estimation_samples=2,
            gamma=10,
            verbose=False,
            num_classes=2,
        )
        img_attacked = attack.generate(img.detach().clone(), (predict > 0.5).long())[0]

    predict_attack = model(img_attacked)
    return (
        img_np, 
        predict.item() > 0.5, 
        _transform_input(img_attacked).detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0), 
        predict_attack.item() > 0.5,
    )

def make_interparetation(
    img_np: np.ndarray, 
    img_attack: np.ndarray, 
    interpretation_name: str,
    model: torch.nn.Module,
    device: str, 
):
    if interpretation_name == "Without interpretation":
        gradcam_img = img_np.copy()
        gradcam_img_attack = img_attack.copy()
    elif interpretation_name == "GradCAM":
        gradcam_img = generate_gradcam(img_np, model, device)
        gradcam_img_attack = generate_gradcam(img_attack, model, device)
    return gradcam_img, gradcam_img_attack

def load_model(
    model_name: str,
) -> torch.nn.Module:
    model_cls = MODEL_OPTIONS[model_name][0]
    model_weights = MODEL_OPTIONS[model_name][1]
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_cls()
    model.load_state_dict(
        torch.load(model_weights, map_location=map_location)
    )
    model = model.to(map_location)

    return model, map_location

def generate_gradcam(
    img_np: np.ndarray,
    model: torch.nn.Module,
    device: str, 
) -> np.ndarray:
    img = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0).to(device)
    layer = [model.model.features[-1]]
    img = _transform_output(img)
    cam = get_gradcam(
        model=model,
        x=img,
        target_layers=layer,
    )
    
    thr = 0.3
    alpha = 0.5
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.clip((cam - thr) / (1 - thr), 0, 1)
    heatmap = plt.cm.jet(cam)[..., :3]

    overlay = img_np.copy()
    mask = cam > 0

    mask = mask.squeeze(0)[..., None]

    overlay = img_np * (1 - alpha * mask) + heatmap * (alpha * mask)
    overlay = np.clip(overlay, 0, 1)

    return overlay.squeeze(0)

def load_selected_input(
    uploaded_file: Optional[str],
    standard_name: Optional[str],
) -> Tuple[Image.Image, str]:
    if uploaded_file:
        img = dicom_to_pil(uploaded_file)
        source_name = Path(uploaded_file).name
        return img, source_name

    if standard_name:
        dicom_path = STANDARD_DICOM_DIR / standard_name
        if not dicom_path.exists():
            raise FileNotFoundError(f"Standard DICOM not found: {dicom_path}")
        img = dicom_to_pil(dicom_path)
        return img, standard_name

    raise ValueError("Please upload a DICOM file or select a standard example.")

def make_result_header(
    pred_label: str = "—",
    model_name: Optional[str] = None,
    attack_name: Optional[str] = None,
    source_name: Optional[str] = None,
    interpretation_name: Optional[str] = None,
) -> str:
    details = ""
    if model_name and attack_name and source_name:
        details = (
            "<div class='result-details'>"
            f"Source: <b>{source_name}</b> · "
            f"Model: <b>{model_name}</b> · "
            f"Corruption type: <b>{attack_name}</b> · "
            f"Interpretation type: <b>{interpretation_name}</b>"
            "</div>"
        )

    return (
        "<div class='result-header'>"
        f"{details}"
        "</div>"
    )

def inference_pipeline(uploaded_file, standard_name, model_name, attack_name, interpretation_name):
    img_pil, source_name = load_selected_input(uploaded_file, standard_name)

    img_rgb = prepare_display_image(img_pil)
    img_np = np.asarray(img_rgb).astype(np.float32) / 255.0

    model, device = load_model(model_name)
    img_np, predict, attacked_np, attacked_predict = apply_attack(img_np, attack_name, model, device)
    gradcam_img, gradcam_img_attack = make_interparetation(img_np, attacked_np, interpretation_name, model, device)

    attacked_img = Image.fromarray((attacked_np * 255).astype(np.uint8))
    img = Image.fromarray((img_np * 255).astype(np.uint8))
    gradcam_img = Image.fromarray((gradcam_img * 255).astype(np.uint8))
    gradcam_img_attack = Image.fromarray((gradcam_img_attack * 255).astype(np.uint8))

    result_header = make_result_header(
        pred_label=predict,
        model_name=model_name,
        attack_name=attack_name,
        source_name=source_name,
        interpretation_name=interpretation_name,
    )

    return (
        result_header, 
        gr.HTML(f"""
        <div style="text-align: center; line-height: 1.3;">
            <div style="font-size: 18px; font-weight: 700;">
                Original image
            </div>
            <div style="font-size: 14px;">
                Model prediction: <b>{predict}</b>
            </div>
        </div>
        """),
        gr.update(
            value=img,
            label=f"Original",
        ),
        gr.HTML(f"""
        <div style="text-align: center; line-height: 1.3;">
            <div style="font-size: 18px; font-weight: 700;">
                Corrupted image
            </div>
            <div style="font-size: 14px;">
                Model prediction: <b>{attacked_predict}</b>
            </div>
        </div>
        """),
        gr.update(
            value=attacked_img,
            label=f"Corrupted",
        ),
        gr.HTML(f"""
        <div style="text-align: center; line-height: 1.3;">
            <div style="font-size: 18px; font-weight: 700;">
                Original image
            </div>
            <div style="font-size: 14px;">
                Model prediction: <b>{predict}</b>
            </div>
        </div> 
        """),
        gr.update(
            value=gradcam_img,
            label=f"{interpretation_name}—Original",
        ),
        gr.HTML(f"""
        <div style="text-align: center; line-height: 1.3;">
            <div style="font-size: 18px; font-weight: 700;"> 
                Corrupted image
            </div>
            <div style="font-size: 14px;">
                Model prediction: <b>{attacked_predict}</b> 
            </div>
        </div>
        """),
        gr.update(
            value=gradcam_img_attack,
            label=f"{interpretation_name}-Corrupted",
        ),
    )

def clear_all():
    standard_dicoms = list_standard_dicoms()
    default_standard = standard_dicoms[0] if standard_dicoms else None

    return (
        None,
        default_standard,
        list(MODEL_OPTIONS.keys())[0],
        "Without corruption",
        "Without interpretation",
        make_result_header(),
        "Results: ",
        gr.update(value=None, label="Original"),
        "Results: ",
        gr.update(value=None, label="Corrupted"),
        "Results: ",
        gr.update(value=None, label="Interpretation-Orig"),
        "Results: ",
        gr.update(value=None, label="Interpretation-Corrupted"),
    )

def build_demo() -> gr.Blocks:
    standard_dicoms = list_standard_dicoms()

    css = """
    .title-block {
        text-align: center;
        margin: 10px 0 25px 0;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #111827;
        margin-bottom: 6px;
    }

    .subtitle {
        font-size: 14px;
        color: #6b7280;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.4;
    }

    .app-wrap {
        max-width: 1500px !important;
        margin: 0 auto;
        gap: 20px;
    }

    /* --- PANELS --- */
    .left-panel {
        border-right: 1px solid #e5e7eb;
        padding: 20px 16px 20px 0;
    }

    .right-panel {
        padding: 20px 0 20px 16px;
    }

    /* --- GLOBAL --- */
    .gradio-container {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        background-color: #fafafa;
    }

    /* --- HEADERS --- */
    .result-header h2 {
        margin: 0 0 8px 0;
        font-size: 22px;
        line-height: 1.3;
        font-weight: 700;
    }

    .result-header h2 span {
        font-weight: 800;
    }

    .result-details {
        margin: 0 0 14px 0;
        font-size: 13px;
        color: #6b7280;
    }

    /* --- INPUT AREA --- */
    #dicom_upload {
        min-height: 260px;
        border: 1px dashed #d1d5db !important;
        border-radius: 10px;
        background: #ffffff;
    }

    /* --- RADIO --- */
    .no-margin {
        margin-bottom: 6px !important;
    }

    /* --- IMAGE BLOCK --- */
    .gradio-row {
        gap: 16px !important;
    }

    .gradio-image {
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid #e5e7eb;
        background: #ffffff;
    }

    /* --- RESULT TEXT --- */
    .markdown p {
        margin: 4px 0;
        font-size: 14px;
    }

    /* --- BUTTONS --- */
    button {
        border-radius: 8px !important;
        font-weight: 500;
    }

    button.primary {
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }

    /* --- DROPDOWNS / INPUTS --- */
    .gradio-dropdown,
    .gradio-file {
        border-radius: 8px !important;
    }

    /* --- SCROLL / SPACING FIX --- */
    .block {
        padding-top: 4px;
        padding-bottom: 4px;
    }
    """

    def on_upload(file):
        if file is not None:
            return gr.update(value=None, interactive=False)
        return gr.update(interactive=True)

    def render_view(
        mode,
        img1_res, img1,
        img2_res, img2,
        img3_res, img3,
        img4_res, img4,
    ):
        if mode == "Original":
            return img1_res, img1, img2_res, img2
        elif mode == "Interpretation":
            return img3_res, img3, img4_res, img4

        return None, None, None, None

    with gr.Blocks(
        css=css,
        title="DICOM Model + Interpretation",
        theme=gr.themes.Origin(),
    ) as demo:
        img1_state = gr.State()
        img2_state = gr.State()
        img3_state = gr.State()
        img4_state = gr.State()

        img1_res_state = gr.State()
        img2_res_state = gr.State()
        img3_res_state = gr.State()
        img4_res_state = gr.State()

        gr.HTML("""
        <div class="title-block">
            <div class="main-title">ClinicalTrustLab</div>
            <div class="subtitle">
                Upload your DICOM file or select a standard example, then choose a model and an corruption type.
            </div>
        </div>
        """)

        with gr.Row(elem_classes=["app-wrap"]):
            with gr.Column(scale=1, min_width=380, elem_classes=["left-panel"]):
                gr.Markdown("## Input")

                upload_file = gr.File(
                    label="Upload your DICOM",
                    file_types=[".dcm", ".dicom", ".DCM"],
                    type="filepath",
                    elem_id="dicom_upload",
                )

                standard_dicom = gr.Dropdown(
                    label="Or select a standard DICOM",
                    choices=standard_dicoms,
                    value=standard_dicoms[0] if standard_dicoms else None,
                    allow_custom_value=False,
                )

                model_selector = gr.Dropdown(
                    label="Model selection",
                    choices=MODEL_OPTIONS.keys(),
                    value=list(MODEL_OPTIONS.keys())[0],
                    allow_custom_value=False,
                )

                attack_selector = gr.Dropdown(
                    label="Corruption selection",
                    choices=ATTACK_OPTIONS,
                    value="Without corruption",
                    allow_custom_value=False,
                )

                interpretation_selector = gr.Dropdown(
                    label="Interpretation selection",
                    choices=INTERPRETATION_OPTIONS,
                    value="Without interpretation",
                    allow_custom_value=False,
                )

                run_button = gr.Button("Run", variant="primary")
                clear_button = gr.Button("Clear")

                gr.Markdown(
                    "### Note\n"
                    "If a custom file is uploaded, it takes priority over the selected standard DICOM."
                )

            with gr.Column(scale=2, min_width=370, elem_classes=["right-panel"]):
                result_header = gr.HTML(make_result_header())

                view_mode = gr.Radio(
                    label="View mode",
                    choices=["Original", "Interpretation"],
                    value="Original",
                    elem_classes=["no-margin"]
                )

                # --- SINGLE IMAGE CONTAINER ---
                with gr.Row():
                    with gr.Column():
                        img_left_res = gr.HTML("""
                        <div style="text-align: center; line-height: 1.3;">
                            <div style="font-size: 18px; font-weight: 700;">
                                Original image
                            </div>
                            <div style="font-size: 14px;">
                                Model prediction:
                            </div>
                        </div>
                        """)
                        img_left = gr.Image(
                            label="Original",
                            height=370,
                            interactive=False,
                        )
                    with gr.Column():
                        img_right_res = gr.HTML("""
                        <div style="text-align: center; line-height: 1.3;">
                            <div style="font-size: 18px; font-weight: 700;">
                                Corrupted image
                            </div>
                            <div style="font-size: 14px;">
                                Model prediction:
                            </div>
                        </div>
                        """)
                        img_right = gr.Image(
                            label="Corrupted",
                            height=370,
                            interactive=False,
                        )
        upload_file.change(
            fn=on_upload,
            inputs=[upload_file],
            outputs=[standard_dicom],
        )

        # --- INFERENCE ---
        run_button.click(
            fn=inference_pipeline,
            inputs=[upload_file, standard_dicom, model_selector, attack_selector, interpretation_selector],
            outputs=[
                result_header,
                img1_res_state, img1_state,
                img2_res_state, img2_state,
                img3_res_state, img3_state,
                img4_res_state, img4_state,
            ],
        )

        # --- INITIAL RENDER AFTER RUN ---
        run_button.click(
            fn=render_view,
            inputs=[
                view_mode,
                img1_res_state, img1_state,
                img2_res_state, img2_state,
                img3_res_state, img3_state,
                img4_res_state, img4_state,
            ],
            outputs=[
                img_left_res, img_left,
                img_right_res, img_right,
            ],
        )

        # --- SWITCH VIEW ---
        view_mode.change(
            fn=render_view,
            inputs=[
                view_mode,
                img1_res_state, img1_state,
                img2_res_state, img2_state,
                img3_res_state, img3_state,
                img4_res_state, img4_state,
            ],
            outputs=[
                img_left_res, img_left,
                img_right_res, img_right,
            ],
        )

        clear_button.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                upload_file,
                standard_dicom,
                model_selector,
                attack_selector,
                interpretation_selector,
                result_header,
                img_left_res,
                img_left,
                img_right_res,
                img_right,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
