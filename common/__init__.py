# Common utilities: skeleton-agnostic chain representation, joint initialization
from .chain_shape import (
    normalize_chain_shape, denormalize_chain_shape, resample_chain,
    extract_chain_descriptor, descriptor_to_vector,
    K_POINTS, DESCRIPTOR_DIM, SHAPE_DIM,
)
from .joint_init import load_init_qpos, save_init_qpos