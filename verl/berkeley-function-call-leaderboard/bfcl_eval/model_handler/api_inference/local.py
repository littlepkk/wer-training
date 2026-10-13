import json
import os
import time
from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from overrides import override
from tenacity import stop_after_attempt


def _is_retryable_xiaoai_error(exc: BaseException) -> bool:
    message = str(exc)
    if "Insufficient Balance" in message:
        return False

    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            json.JSONDecodeError,
        ),
    ):
        return True

    if isinstance(exc, APIStatusError):
        status_code = exc.status_code
        return status_code in {408, 409, 429} or status_code >= 500

    return False


class localAPIHandler(OpenAICompletionsHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        base = os.getenv("WER_EXECUTOR_BASE_URL", "http://localhost:8000/v1")

        self.client = OpenAI(
            base_url=base,
            api_key=os.getenv("WER_EXECUTOR_API_KEY", "EMPTY"),
            timeout=360.0,
            max_retries=0,
        )

    @retry_with_backoff(
        retry_condition=_is_retryable_xiaoai_error,
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def generate_with_backoff(self, **kwargs):
        """
        XiaoAI network calls can fail transiently because of connection issues,
        timeouts, rate limits, or upstream 5xx errors. The retry condition keeps
        those retryable while letting non-recoverable request/auth errors fail fast.
        """
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        if len(tools) > 0:
            return self.generate_with_backoff(
                model=self.model_name,
                messages=message,
                tools=tools,
                temperature=self.temperature,
            )
        else:
            return self.generate_with_backoff(
                model=self.model_name,
                messages=message,
                temperature=self.temperature,
            )

    @override
    def _query_prompting(self, inference_data: dict):
        """
        This method is intended to be used by the `DeepSeek-R1` models. If used for other models, you will need to modify the code accordingly.

        Reasoning models don't support temperature parameter
        https://api-docs.deepseek.com/guides/reasoning_model

        `DeepSeek-R1` should use `deepseek-reasoner` as the model name in the API
        https://api-docs.deepseek.com/quick_start/pricing
        """
        message: list[dict] = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(message)}

        return self.generate_with_backoff(
            model=self.model_name,
            messages=message,
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        # 'deepseek-reasoner does not support successive user messages, so we need to combine them
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        return {"message": []}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        DeepSeek does not take reasoning content in next turn chat history, for both prompting and function calling mode.
        Error: Error code: 400 - {'error': {'message': 'The reasoning_content is an intermediate result for display purposes only and will not be included in the context for inference. Please remove the reasoning_content from your message to reduce network traffic.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
        """
        response_data = super()._parse_query_response_prompting(api_response)
        self._add_reasoning_content_if_available_prompting(api_response, response_data)
        return response_data

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        DeepSeek does not take reasoning content in next turn chat history, for both prompting and function calling mode.
        Error: Error code: 400 - {'error': {'message': 'The reasoning_content is an intermediate result for display purposes only and will not be included in the context for inference. Please remove the reasoning_content from your message to reduce network traffic.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
        """
        response_data = super()._parse_query_response_FC(api_response)
        self._add_reasoning_content_if_available_FC(api_response, response_data)
        return response_data
