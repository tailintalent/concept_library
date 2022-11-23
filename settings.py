import torch

REPR_DIM = 8
DEFAULT_OBJ_TYPE = "Image"
DEFAULT_SYS_TYPE = "System"
DEFAULT_BODY_TYPE = "Body"
RELATION_EMBEDDING = {
    "changeColor": torch.nn.Parameter(torch.rand(REPR_DIM)),
}