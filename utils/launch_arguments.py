from pydantic import BaseModel, Field

class LaunchArguments(BaseModel):
    share: bool = Field(False, description="Set whether to share on Gradio.")
    preset: str = Field(None, description="Apply specified UI preset.")
    disable_preset_selection: bool = Field(False, description="Disables preset selection in Gradio.")
    language: str = Field(None, description="Translate UI using json files in [language] folder.")
    disable_offload_from_vram: bool = Field(False, description="Force loading models to vram when the unload can be avoided.")
    theme: str = Field(None, description="Launches the UI with light or dark theme.")
    disable_image_log: bool = Field(False, description="Prevent writing images and logs to the outputs folder.")
    disable_analytics: bool = Field(False, description="Disables analytics for Gradio.")
    disable_metadata: bool = Field(False, description="Disables saving metadata to images.")
    disable_preset_download: bool = Field(False, description="Disables downloading models for presets.")
    disable_enhance_output_sorting: bool = Field(False, description="Disables enhance output sorting for final image gallery.")
    enable_auto_describe_image: bool = Field(False, description="Enables automatic description of uov and enhance image when prompt is empty.")
    always_download_new_model: bool = Field(False, description="Always download newer models.")
    rebuild_hash_cache: bool = Field(False, description="Generates missing model and LoRA hashes.")
    headless: bool = Field(False, description="Run in headless mode.")


args = LaunchArguments()