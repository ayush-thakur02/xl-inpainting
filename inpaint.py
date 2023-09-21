# Github: ayush-thakur02
# Bio: bio.link/ayush_thakur02
# Its's an Open Source Project, make sure to share with others and give it a star :)

import gradio as gr
import torch

from diffusers import AutoPipelineForInpainting, UNet2DConditionModel
import diffusers

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)

def read_content(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def predict(dict, prompt="", negative_prompt="", guidance_scale=8.5, steps=25, strength=1.0, scheduler="EulerDiscreteScheduler"):
    if negative_prompt == "":
        negative_prompt = None
    scheduler_class_name = scheduler.split("-")[0]

    add_kwargs = {}
    if len(scheduler.split("-")) > 1:
        add_kwargs["use_karras"] = True
    if len(scheduler.split("-")) > 2:
        add_kwargs["algorithm_type"] = "sde-dpmsolver++"

    scheduler = getattr(diffusers, scheduler_class_name)
    pipe.scheduler = scheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **add_kwargs)

    init_image = dict["image"].convert("RGB").resize((1024, 1024))
    mask = dict["mask"].convert("RGB").resize((1024, 1024))

    output = pipe(prompt = prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask, guidance_scale=guidance_scale, num_inference_steps=int(steps), strength=strength)

    return output.images[0], gr.update(visible=True)

# CSS for the Web UI Elements
css = '''
.gradio-container{max-width: 1100px !important}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#prompt input{width: calc(100% - 160px);border-top-right-radius: 0px;border-bottom-right-radius: 0px;}
#run_button{position:absolute;margin-top: 11px;right: 0;margin-right: 0.8em;border-bottom-left-radius: 0px;
    border-top-left-radius: 0px;}
#prompt-container{margin-top:-18px;}
#prompt-container .form{border-top-left-radius: 0;border-top-right-radius: 0}
#image_upload{border-bottom-left-radius: 0px;border-bottom-right-radius: 0px}
'''

image_blocks = gr.Blocks(css=css, elem_id="total-container")
with image_blocks as demo:
    with gr.Row():
                with gr.Column():
                    image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload",height=400)
                    with gr.Row(elem_id="prompt-container", mobile_collapse=False, equal_height=True):
                        with gr.Row():
                            prompt = gr.Textbox(placeholder="Your prompt - What you want in place of what is erased", show_label=False, elem_id="prompt")
                            btn = gr.Button("Inpaint!", elem_id="run_button")

                    with gr.Accordion(label="Advanced Settings", open=False):
                        with gr.Row(mobile_collapse=False, equal_height=True):
                            guidance_scale = gr.Number(value=8.5, minimum=1.0, maximum=25.0, step=0.1, label="guidance_scale")
                            steps = gr.Number(value=20, minimum=10, maximum=30, step=1, label="steps")
                            strength = gr.Number(value=0.99, minimum=0.01, maximum=0.99, step=0.01, label="strength")
                            negative_prompt = gr.Textbox(label="negative_prompt", placeholder="Your negative prompt", info="what you don't want to see in the image")
                        with gr.Row(mobile_collapse=False, equal_height=True):
                            schedulers = ["DEISMultistepScheduler", "HeunDiscreteScheduler", "EulerDiscreteScheduler", "DPMSolverMultistepScheduler", "DPMSolverMultistepScheduler-Karras", "DPMSolverMultistepScheduler-Karras-SDE"]
                            scheduler = gr.Dropdown(label="Schedulers", choices=schedulers, value="EulerDiscreteScheduler")

                with gr.Column():
                    image_out = gr.Image(label="Output", elem_id="output-img", height=400)

    btn.click(fn=predict, inputs=[image, prompt, negative_prompt, guidance_scale, steps, strength, scheduler], outputs=[image_out], api_name='run')
    prompt.submit(fn=predict, inputs=[image, prompt, negative_prompt, guidance_scale, steps, strength, scheduler], outputs=[image_out])
image_blocks.queue(max_size=25).launch(debug=True, share=True)
