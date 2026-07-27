import inspect
import logging
import sys

from .clip import CLIPEncoder
from .convnext import ConvNeXTEncoder
from .data2vec import HuggingFaceTransformerEncoder
from .dinov2 import DINOv2Encoder
from .encoder import Encoder
from .inception import InceptionEncoder
from .mae import VisionTransformerEncoder
from .pixel import PixelEncoder
from .simclr import SimCLRResNetEncoder
from .swav import ResNet50Encoder  # ResNet18Encoder, ResNet18MocoEncoder,

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)

MODEL_ENCODERS = {
    "inception": InceptionEncoder,
    "sinception": InceptionEncoder,
    "mae": VisionTransformerEncoder,
    "data2vec": HuggingFaceTransformerEncoder,
    "swav": ResNet50Encoder,
    "clip": CLIPEncoder,
    "pixel": PixelEncoder,
    "convnext": ConvNeXTEncoder,
    "dinov2": DINOv2Encoder,
    "simclr": SimCLRResNetEncoder,
}

MODELS = list(MODEL_ENCODERS.keys())


def load_encoder(model_name, device, **kwargs):
    """Load feature extractor"""
    logger.info("Loading Model...")

    model_cls = MODEL_ENCODERS.get(model_name)

    # Get names of model_cls.setup arguments
    signature = inspect.signature(model_cls.setup)
    arguments = list(signature.parameters.keys())
    arguments = arguments[1:]  # Omit `self` arg

    # Initialize model using the `arguments` that have been passed in the `kwargs` dict
    encoder = model_cls(**{arg: kwargs[arg] for arg in arguments if arg in kwargs})
    encoder.name = model_name

    assert isinstance(encoder, Encoder), (
        "Can only get representations with Encoder subclasses!"
    )

    return encoder.to(device)
