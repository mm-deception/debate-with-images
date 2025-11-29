"""
General utility function module
Centralized import of all functional modules, providing unified interface
"""

# Import all functions from specialized modules
from api_utils import (
    bean_gpt_api,
    generate_hash_uid,
    api,
    vllm_api
)

from format_utils import (
    load_debate_ds,
    build_inference_payload,
    clear_visual_cache,
    build_init_user_prompt,
    format_history_item,
    extract_json_blocks
)

from visual_utils import (
    plot_bounding_boxes,
    encode_image_to_base64,
    compress_image,
    combine_images
)

# For backward compatibility, keep the original import method
__all__ = [
    # API related
    'bean_gpt_api',
    'generate_hash_uid', 
    'api',
    'vllm_api',
    
    # Formatting related
    'load_debate_ds',
    'build_inference_payload',
    'clear_visual_cache',
    'build_init_user_prompt',
    'format_history_item',
    'extract_json_blocks',
    
    # Visualization related
    'plot_bounding_boxes',
    'combine_images',
    'compress_image',
    'encode_image_to_base64',
]
