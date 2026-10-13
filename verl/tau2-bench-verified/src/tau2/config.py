import os

from dotenv import load_dotenv


# Load project-local credentials before defining configuration constants.
load_dotenv()


# SIMULATION
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "ERROR"

# LLM
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = os.getenv("TAU2_AGENT_MODEL", f"openai/{os.getenv('WER_EXECUTOR_MODEL', 'executor-model')}")
DEFAULT_LLM_USER = os.getenv("TAU2_USER_MODEL", "openai/user-simulator-model")
DEFAULT_LLM_TEMPERATURE_AGENT = 0.0
DEFAULT_LLM_TEMPERATURE_USER = 0.0
DEFAULT_LLM_ARGS_AGENT = {"temperature": DEFAULT_LLM_TEMPERATURE_AGENT}
DEFAULT_LLM_ARGS_USER = {"temperature": DEFAULT_LLM_TEMPERATURE_USER}

DEFAULT_LLM_NL_ASSERTIONS = os.getenv("TAU2_NL_ASSERTIONS_MODEL", "openai/judge-model")
DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE = 0.0
DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE}

DEFAULT_LLM_ENV_INTERFACE = os.getenv("TAU2_ENV_INTERFACE_MODEL", "openai/environment-interface-model")
DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE = 0.0
DEFAULT_LLM_ENV_INTERFACE_ARGS = {"temperature": DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE}

# LITELLM
DEFAULT_MAX_RETRIES = 3
LLM_CACHE_ENABLED = False
DEFAULT_LLM_CACHE_TYPE = "redis"

# Optional global OpenAI-compatible endpoint. These values are injected into
# every LiteLLM completion unless that individual call explicitly supplies
# api_base/api_key in its LLM arguments.
TAU2_LLM_API_BASE = os.getenv("TAU2_LLM_API_BASE") or None
TAU2_LLM_API_KEY = os.getenv("TAU2_LLM_API_KEY") or None

# REDIS CACHE
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""
REDIS_PREFIX = "tau2"
REDIS_CACHE_VERSION = "v1"
REDIS_CACHE_TTL = 60 * 60 * 24 * 30

# LANGFUSE
USE_LANGFUSE = False  # If True, make sure all the env variables are set for langfuse.

# API
API_PORT = 8000
