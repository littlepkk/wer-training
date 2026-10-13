import os
from dataclasses import dataclass
from typing import Optional

from bfcl_eval.model_handler.api_inference.local import localAPIHandler


@dataclass
class ModelConfig:
    """
    Model configuration class for storing model metadata and settings.

    Attributes:
        model_name (str): Name of the model as used in the vendor API or on Hugging Face (may not be unique).
        display_name (str): Model name as it should appear on the leaderboard.
        url (str): Reference URL for the model or hosting service.
        org (str): Organization providing the model.
        license (str): License under which the model is released.
        model_handler (str): Handler name for invoking the model.
        input_price (Optional[float]): USD per million input tokens (None for open source models).
        output_price (Optional[float]): USD per million output tokens (None for open source models).
        is_fc_model (bool): True if this model is used in Function-Calling mode, otherwise False for Prompt-based mode.
        underscore_to_dot (bool): True if model does not support '.' in function names, in which case we will replace '.' with '_'. Currently this only matters for checker.  TODO: We should let the tool compilation step also take this into account.

    """

    model_name: str
    display_name: str
    url: str
    org: str
    license: str

    model_handler: str

    # Prices are in USD per million tokens; open source models have None
    input_price: Optional[float] = None
    output_price: Optional[float] = None

    # True if the model is in function-calling mode, False if in prompt mode
    is_fc_model: bool = True

    # True if this model does not allow '.' in function names
    underscore_to_dot: bool = False


MODEL_CONFIG_MAPPING = {
    "local_fc": ModelConfig(
        model_name=os.getenv("WER_EXECUTOR_MODEL", "executor-model"),
        display_name="Training executor",
        url=os.getenv("WER_EXECUTOR_BASE_URL", "http://localhost:8000/v1"),
        org="",
        license="",
        model_handler=localAPIHandler,
        is_fc_model=True,
        underscore_to_dot=True,
    ),
}
