import gradio as gr
import torch
import os

from transformers import AutoTokenizer, AutoFeatureExtractor

from tts_webui.utils.get_path_from_root import get_path_from_root
from tts_webui.decorators.gradio_dict_decorator import dictionarize
from tts_webui.utils.randomize_seed import randomize_seed_ui
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.list_dir_models import model_select_ui, unload_model_button
from tts_webui.decorators.decorator_apply_torch_seed import decorator_apply_torch_seed
from tts_webui.decorators.decorator_log_generation import decorator_log_generation
from tts_webui.decorators.decorator_save_metadata import decorator_save_metadata
from tts_webui.decorators.decorator_save_wav import decorator_save_wav
from tts_webui.decorators.decorator_add_base_filename import decorator_add_base_filename
from tts_webui.decorators.decorator_add_date import decorator_add_date
from tts_webui.decorators.decorator_add_model_type import decorator_add_model_type
from tts_webui.decorators.log_function_time import log_function_time

from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_outer,
    decorator_extension_inner,
)


def extension__tts_generation_webui():
    main_ui()
    return {
        "package_name": "extension_audio_separator",
        "name": "Audio Separator",
        "version": "0.0.1",
        "requirements": "git+https://github.com/rsxdalv/extension_audio_separator@main",
        "description": "Audio Separator allows separating audio files into multiple audio files.",
        "extension_type": "interface",
        "extension_class": "audio-conversion",
        "author": "rsxdalv",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/rsxdalv/extension_audio_separator",
        "extension_website": "https://github.com/rsxdalv/extension_audio_separator",
        "extension_platform_version": "0.0.1",
    }


device = "cuda:0" if torch.cuda.is_available() else "cpu"


repo_id = "parler-tts/parler-tts-mini-v1"
repo_id_large = "ylacombe/parler-large-v1-og"

feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
SAMPLE_RATE = feature_extractor.sampling_rate

LOCAL_DIR = os.path.join("data", "models", "parler_tts")
LOCAL_MODEL_DIR = os.path.join(LOCAL_DIR, "cache")


@manage_model_state(model_namespace="parler_tts")
def get_parler_tts_model(
    model_name=repo_id, attn_implementation=None, compile_mode=None
):
    from parler_tts import ParlerTTSForConditionalGeneration

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=LOCAL_MODEL_DIR,
        attn_implementation=attn_implementation,
        # attn_implementation = "eager" # "sdpa" or "flash_attention_2"
    ).to(device)

    if compile_mode is not None:
        # compile_mode = "default"  # chose "reduce-overhead" for 3 to 4x speed-up
        model.generation_config.cache_implementation = "static"
        # compile the forward pass
        model.forward = torch.compile(model.forward, mode=compile_mode)

    return model


@manage_model_state(model_namespace="parler_tts_tokenizer")
def get_tokenizer(model_name=repo_id):
    return AutoTokenizer.from_pretrained(model_name, cache_dir=LOCAL_MODEL_DIR)


@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("parler_tts")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def generate_parler_tts(
    text,
    description,
    model_name,
    attn_implementation=None,
    compile_mode=None,
    **kwargs,
):
    tokenizer = get_tokenizer(repo_id)
    inputs = tokenizer(description.strip(), return_tensors="pt").to(device)
    prompt = tokenizer(text, return_tensors="pt").to(device)

    model = get_parler_tts_model(
        model_name, attn_implementation=attn_implementation, compile_mode=compile_mode
    )

    generation = model.generate(
        input_ids=inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_attention_mask=prompt.attention_mask,
        do_sample=True,
        temperature=1.0,
    )

    return {"audio_out": (SAMPLE_RATE, generation.squeeze(0).cpu().numpy())}


def parler_tts_params_ui():
    text = gr.Textbox(
        label="Text",
        value="Hey, how are you doing today?",
    )
    description = gr.Textbox(
        label="Context",
        value="A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    )

    return {text: "text", description: "description"}


def main_ui():
    gr.Markdown(
        """
    This extension is not yet available as an UI, it can only be used with command line.

    To use it, simply run the following command inside of the tts-generation-webui/installer_scripts/conda_env_cmd.bat terminal:

    ```
    audio-separator file.mp3 --model_filename UVR-MDX-NET-Inst_HQ_3.onnx
    ```
    """
    )
    return

    gr.Markdown(
        """
        # Parler-TTS
        Parler-TTS is a training and inference library for high-fidelity text-to-speech (TTS) models.

        
        More models can be found at: https://huggingface.co/models?filter=parler_tts
        """
    )

    with gr.Column():
        with gr.Column():
            parler_tts_params = parler_tts_params_ui()

            with gr.Row():
                model_name = model_select_ui(
                    [
                        ("Parler-TTS Mini v1", repo_id),
                        ("Parler-TTS Large v1", repo_id_large),
                    ],
                    "parler_tts",
                )
                seed, randomize_seed_callback = randomize_seed_ui()

            attn_implementation = gr.Dropdown(
                choices=["eager", "sdpa", "flash_attention_2"],
                label="Attention Implementation",
                value="sdpa",
            )

            compile_mode = gr.Dropdown(
                choices=[("None", None), "default", "reduce-overhead"],
                label="Compile Mode",
                value=None,
            )

            unload_model_button("parler_tts")

        with gr.Column():
            audio_out = gr.Audio(
                label="Parler-TTS generation", type="numpy", elem_id="audio_out"
            )

    gr.Button("Generate Audio", variant="primary").click(
        **randomize_seed_callback
    ).then(
        **dictionarize(
            fn=generate_parler_tts,
            inputs={
                **parler_tts_params,
                seed: "seed",
                model_name: "model_name",
                attn_implementation: "attn_implementation",
                compile_mode: "compile_mode",
            },
            outputs={"audio_out": audio_out},
        ),
        api_name="parler_tts",
    )


if __name__ == "__main__":
    if "demo" in locals():
        demo.close()

    with gr.Blocks() as demo:
        main_ui()

    demo.launch()
