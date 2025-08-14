from hydra.core.global_hydra import GlobalHydra
from hydra.types import RunMode

# Get the global hydra instance
hydra = GlobalHydra.instance().hydra
if hydra is None:
    raise RuntimeError("Hydra is not initialized")


# Get the composed config
cfg = hydra.compose_config(
    config_name="cfg_pretrain",
    overrides=[],
    run_mode=RunMode.MULTIRUN
)

# Return just the attn value
module_path, class_name = cfg.get("arch", {}).get("attn").split('.')

print(f"module_path={module_path}, class_name={class_name}")

# class_name would be default, rwkv, linear
# if class_name == 'linear':
    # from linear_transformer_attn import Attention
# elif class_name == 'rwkv':
from .rwkv_attn import RWKVAttention as Attention
# else:
# from .pytorch_or_flash_basic_attn import Attention

