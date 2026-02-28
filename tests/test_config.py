import os
import sys
import types
import importlib


def _setup_stubs():
    """Stub pydantic-settings so config module can be imported without it."""
    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

    pydantic.BaseModel = BaseModel
    sys.modules["pydantic"] = pydantic

    pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        """Minimal BaseSettings replacement that reads env vars."""

        model_config = {}

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

        def __init__(self, **overrides):
            prefix = getattr(
                getattr(self.__class__, "model_config", {}),
                "get",
                lambda k, d="": (self.__class__.model_config or {}).get(k, d),
            )("env_prefix", "")
            # If model_config is a plain dict, read prefix from it
            if isinstance(self.__class__.model_config, dict):
                prefix = self.__class__.model_config.get("env_prefix", "")

            # Walk class annotations for defaults, then apply env overrides
            import typing

            for attr_name in typing.get_type_hints(self.__class__):
                if attr_name == "model_config":
                    continue
                # Start with the class-level default
                default = getattr(self.__class__, attr_name, None)
                # Check environment variable
                env_key = f"{prefix}{attr_name}".upper()
                env_val = os.environ.get(env_key)
                if env_val is not None:
                    # Coerce to the expected type
                    hint = typing.get_type_hints(self.__class__)[attr_name]
                    if hint is bool:
                        default = env_val.lower() in ("true", "1", "yes")
                    elif hint is int:
                        default = int(env_val)
                    elif hint is float:
                        default = float(env_val)
                    else:
                        default = env_val
                # Allow explicit overrides
                if attr_name in overrides:
                    default = overrides[attr_name]
                setattr(self, attr_name, default)

    pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = pydantic_settings


def _clear_modules():
    for mod in list(sys.modules):
        if "diarized_transcriber" in mod:
            del sys.modules[mod]


def test_default_settings():
    """ServiceSettings has sensible defaults."""
    _setup_stubs()
    _clear_modules()

    from pathlib import Path

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [
        str(Path(__file__).resolve().parents[1] / "src" / "diarized_transcriber")
    ]
    sys.modules["diarized_transcriber"] = pkg

    config_mod = importlib.import_module("diarized_transcriber.config")
    # Clear lru_cache so we get fresh settings
    config_mod.get_settings.cache_clear()

    settings = config_mod.ServiceSettings()
    assert settings.redis_url == "redis://localhost:6379"
    assert settings.model_size == "large-v3-turbo"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.idle_timeout == 300
    assert settings.extract_embeddings is True
    assert settings.speaker_db_path == "./speaker_profiles_db"
    assert settings.identify_speakers is False
    assert settings.auto_enroll_speakers is False
    assert settings.match_threshold == 0.4

    _clear_modules()


def test_env_var_overrides():
    """Environment variables with DIARIZED_ prefix override defaults."""
    _setup_stubs()
    _clear_modules()

    from pathlib import Path

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [
        str(Path(__file__).resolve().parents[1] / "src" / "diarized_transcriber")
    ]
    sys.modules["diarized_transcriber"] = pkg

    config_mod = importlib.import_module("diarized_transcriber.config")
    config_mod.get_settings.cache_clear()

    env_overrides = {
        "DIARIZED_PORT": "9999",
        "DIARIZED_MODEL_SIZE": "tiny",
        "DIARIZED_EXTRACT_EMBEDDINGS": "false",
        "DIARIZED_MATCH_THRESHOLD": "0.7",
    }
    original = {}
    for k, v in env_overrides.items():
        original[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        settings = config_mod.ServiceSettings()
        assert settings.port == 9999
        assert settings.model_size == "tiny"
        assert settings.extract_embeddings is False
        assert settings.match_threshold == 0.7
    finally:
        for k, orig in original.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig

    _clear_modules()


def test_get_settings_caching():
    """get_settings() returns the same object on repeated calls."""
    _setup_stubs()
    _clear_modules()

    from pathlib import Path

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [
        str(Path(__file__).resolve().parents[1] / "src" / "diarized_transcriber")
    ]
    sys.modules["diarized_transcriber"] = pkg

    config_mod = importlib.import_module("diarized_transcriber.config")
    config_mod.get_settings.cache_clear()

    s1 = config_mod.get_settings()
    s2 = config_mod.get_settings()
    assert s1 is s2

    _clear_modules()
