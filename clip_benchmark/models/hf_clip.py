from transformers import CLIPConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer
from functools import partial

class HFCLIPModel(CLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)

    def encode_image(self, image):
        return self.get_image_features(**image)

    def encode_text(self, text):
        return self.get_text_features(**text)


def load_hf_clip(pretrained: str, device="cpu", **kwargs):
    model = HFCLIPModel.from_pretrained(pretrained)
    model.to(device)
    processor = CLIPImageProcessor.from_pretrained(pretrained)
    processor = partial(processor, return_tensors="pt")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained)
    tokenizer = partial(tokenizer, padding = True, return_tensors = "pt")

    return model, processor, tokenizer
