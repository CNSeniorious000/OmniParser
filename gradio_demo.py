import base64
import io
from functools import cache
from itertools import accumulate
from typing import Literal

import gradio as gr
from PIL import Image


@cache
def get_yolo_model():
    from utils import get_yolo_model

    return get_yolo_model(model_path="weights/icon_detect/best.pt")


@cache
def get_caption_model_processor(model: Literal["florence2", "blip2"]):
    from utils import get_caption_model_processor

    if model == "florence2":
        return get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

    return get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")


MARKDOWN = """
<h2 style="margin: 1em 0; padding: 0.5em 0; text-align: center"> OmniParser Based UI Agent Demo </h2>
"""


# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    ocr_model: Literal["easyocr", "paddleocr"],
    som_model: Literal["florence2", "blip2"],
    imgsz: int,
    prompt: str,
):
    image_save_path = "imgs/saved_image_demo.png"
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = min(image.size) / 900
    draw_bbox_config = {
        "text_scale": max(0.8 * box_overlay_ratio, 1),
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(4 * box_overlay_ratio), 4),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }
    from utils import check_ocr_box, get_som_labeled_img

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=ocr_model == "paddleocr",
    )
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path,
        get_yolo_model(),
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=get_caption_model_processor(som_model),
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print("finish processing")
    parsed_content_list = "\n".join(parsed_content_list)

    yield image, parsed_content_list, None

    if prompt:
        from chat import ask_llm

        for res in accumulate(ask_llm(prompt, image, parsed_content_list)):
            yield image, parsed_content_list, res


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            som_model_component = gr.Radio(label="SOM Model", choices=["florence2", "blip2"], value="blip2")
            ocr_model_component = gr.Radio(label="OCR Model", choices=["easyocr", "paddleocr"], value="paddleocr")

            # set the threshold for removing the bounding boxes with low confidence, default is 0.05
            box_threshold_component = gr.Slider(label="Box Threshold", minimum=0.01, maximum=1.0, step=0.01, value=0.05)

            # set the threshold for removing the bounding boxes with large overlap, default is 0.1
            iou_threshold_component = gr.Slider(label="IOU Threshold", minimum=0.01, maximum=1.0, step=0.01, value=0.1)

            imgsz_component = gr.Slider(label="Icon Detect Image Size", minimum=640, maximum=1920, step=32, value=640)

            submit_button_component = gr.Button(value="Submit", variant="primary")

            # with gr.Column():
            prompt_input_component = gr.Textbox(label="Prompt", placeholder="Find the 'Add' button")
            image_input_component = gr.Image(type="pil", label="Upload image")

        with gr.Column():
            image_output_component = gr.Image(type="pil", label="Image Output")
            text_output_component = gr.Textbox(label="Parsed screen elements", placeholder="Text Output")
            llm_output_component = gr.Markdown(label="LLM Result", container=True)

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            ocr_model_component,
            som_model_component,
            imgsz_component,
            prompt_input_component,
        ],
        outputs=[image_output_component, text_output_component, llm_output_component],
    )

# demo.launch(debug=False, show_error=True, share=True)
demo.launch(share=True, server_name="0.0.0.0")
