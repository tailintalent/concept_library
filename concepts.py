#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import OrderedDict, Counter
from copy import deepcopy
import itertools
import json
import matplotlib.pylab as plt
import networkx as nx
from numbers import Number
from networkx.algorithms import isomorphism
from networkx.readwrite import json_graph
from networkx import DiGraph
from networkx import line_graph
from networkx.classes.reportviews import NodeView
import numpy as np
import pdb
from scipy import optimize
import torch
import torch.nn as nn
from kmeans_pytorch import kmeans

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from concept_library.settings import REPR_DIM, DEFAULT_OBJ_TYPE, DEFAULT_BODY_TYPE, RELATION_EMBEDDING
from concept_library.models import get_neg_mask_overlap, get_pixel_entropy, get_pixel_gm, get_graph_energy, GNN_energy
from concept_library.util import get_connected_loss, visualize_dataset, visualize_matrices, combine_pos, accepts, broadcast_inputs, get_op_shape, action_equal, get_patch, set_patch, find_connected_components, find_connected_components_colordiff, get_op_type, persist_hash
from concept_library.util import Combined_Dict, Shared_Param_Dict, repeat_n, tensor_to_string, get_inherit_modes, canonical, canonicalize_keys, masked_equal, get_attr_proper_name, combine_dicts, check_result_true, find_valid_operators, get_obj_bounding_pos, get_comp_obj, get_pos_intersection, shrink, get_repr_dict
from concept_library.util import get_soft_Jaccard_distance, to_np_array, to_Variable, plot_matrices, make_dir, remove_duplicates, broadcast_keys, to_string, check_same_set, ddeepcopy as deepcopy
from concept_library.util import COLOR_LIST, get_pdict, copy_with_model_dict, record_data, canonicalize_strings, split_string, get_generalized_mean, try_eval, get_rename_mapping, get_next_available_key

# Node types:
ATTR_TYPE = "attr"        # attribute node in Graph class or Concept class
OBJ_TYPE = "obj"          # Object node in Concept class for a candidate segmentation of an object.
INPUT_TYPE = "input"      # input node in Graph class
INNODE_TYPE = "fun-in"    # in_node in Graph class
OUTNODE_TYPE = "fun-out"  # out_node in Graph class
OPERATOR_TYPE = "self"    # operator node in Graph class
CONCEPT_TYPE = "concept"  # concept node in Concept class, or a constant concept in an operator graph

# Edge types:
OPERATOR_INTRA_EDGE = "intra"         # Edge between operator node and its in_node or out_node
OPERATOR_INTER_EDGE = "inter-input"   # Edge connecting from an out_node (input, attr or fun-out) to an in_node
OPERATOR_CONTROL_EDGE = "inter-ctrl"  # Edge connecting from an in_node to the Ctrl node of a goal operator
GET_ATTR_EDGE = "intra-attr"          # Edge from a concept to its attributes.
RELATION_EDGE = "relation"

# Concept Library:
CONCEPTS = OrderedDict()     # Predefined concepts
NEW_CONCEPTS = OrderedDict() # Newly learned concepts
OPERATORS = OrderedDict()    # Predefined operators

IS_VIEW = True    # Whether to draw graph when printing in ipynb


# ## 1.1 Placeholder and Basic functions:

# ### 1.1.1 Placeholder

# In[ ]:


class Placeholder(object):
    """Placeholder class. Holds a Tensor or a concept string."""
    def __init__(
        self,
        mode,
        name=None,
        value=None,
        shape=None,
        range=None,
        selector=None,
        ebm_key=None,
        inplace=True,
    ):
        """
        Args:
            mode: type of the concept.
            value: value held by the placeholder
            shape: required shape of the placeholder.
            range: range of the placeholder's value
            selector: if not None, will have a selector.
            ebm_key:  if not None, will be the key that points to an EBM in the ebm_dict.
            inplace: whether the selector is inplace. If inplace=True, then the operator will copy other parts
                not selected by the selector to the output. If inplace=False, then the operator's output will
                only consist of the operated objects selected by the selector.
        """
        assert isinstance(mode, Tensor) or isinstance(mode, str), "mode must be a Tensor or a concept string"
        self.mode = mode
        self.value = value
        self.shape = shape
        self.range = range
        self.selector = selector
        self.ebm_key = ebm_key
        self.inplace = inplace


    def __repr__(self):
        if self.value is not None:
            string = "value->"
        else:
            string = ""
        return "Placeholder({}{})".format(string, self.mode)


    def __bool__(self):
        if (self.mode == "Bool" or self.mode.dtype == "bool") and not self.value:
            return False
        else:
            return True


    def change_mode(self, new_mode, new_ebm_key=None):
        """Change the placeholder's mode to a new mode."""
        if new_mode != self.mode:
            self.mode = new_mode
            if hasattr(self, "ebm_key"):
                assert new_ebm_key is not None
                self.ebm_key = new_ebm_key
        return self


    def set_value(self, value):
        self.value = value


    def set_selector(self, selector):
        self.selector = selector


    def get_selector(self):
        return self.selector


    def set_ebm_key(self, ebm_key):
        self.ebm_key = ebm_key
        return self


    def get_ebm_key(self):
        return self.ebm_key


    def set_inplace(self, inplace):
        self.inplace = inplace


    def get_inplace(self):
        return self.inplace
    
    
    def copy_with_grad(self, is_copy_module=True, global_attrs=None):
        copied_dict = {}
        copied_placeholder = Placeholder(self.mode)
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                if value.requires_grad:
                    # Copy tensor, detach it from the original computation graph, then allow gradients again
                    copied_dict[name] = value.clone().detach().requires_grad_()
                else:
                    copied_dict[name] = value.clone()
            elif isinstance(value, Concept):
                copied_dict[name] = value.copy_with_grad(is_copy_module=is_copy_module, global_attrs=global_attrs)
            else:
                copied_dict[name] = deepcopy(value)
        copied_placeholder.__dict__.update(copied_dict)
        return copied_placeholder


    def is_valid(self, input):
        """Check if the input is valid for the current placeholder."""
        if isinstance(self.mode, str):
            is_valid, err = input.name == self.mode, "concept-type"
        else:
            is_valid, err = self.mode.is_valid(input)
        return is_valid, err


    def accepts(self, placeholder, node_name=None):
        """Whether the current placeholder accepts input from the given placeholder."""
        if self.mode == "Concept":
            if isinstance(placeholder, Concept):
                return True
            else:
                return False
        if isinstance(placeholder, Concept):
            node_name = placeholder.name
        else:
            if not isinstance(placeholder, Placeholder):
                return False
        if node_name is not None:
            # the self placeholder accepts a full node:
            if node_name.startswith(self.mode):
                return True
        if self.mode in ["Args", "Ctrl"]:
            # Input nodes for Cri (criteria):
            return True
        else:
            # Grounding by providing input:
            if isinstance(self.mode, str):
                if not isinstance(placeholder.mode, str):
                    return False
                else:
                    if canonical(self.mode) == canonical(placeholder.mode):
                        return True
                    else:
                        # Check if it is inherit from the concept:
                        if canonical(placeholder.mode) in CONCEPTS and hasattr(CONCEPTS[canonical(placeholder.mode)], "inherit_from") and canonical(self.mode) in CONCEPTS[canonical(placeholder.mode)].inherit_from:
                            return True
                        elif canonical(placeholder.mode) in NEW_CONCEPTS and hasattr(NEW_CONCEPTS[canonical(placeholder.mode)], "inherit_from") and canonical(self.mode) in NEW_CONCEPTS[canonical(placeholder.mode)].inherit_from:
                            return True
                        else:
                            return False
            else:
                return self.mode.dtype == placeholder.mode.dtype


# Core classes:
class Tensor(object):
    """Tensor class."""
    def __init__(
        self,
        dtype,
        shape=None,
        range=None,
        **kwargs
    ):
        assert dtype in ["cat", "bool", "N", "real"]
        self.dtype = dtype
        self.shape = shape
        self.range = range
        for key, item in kwargs.items():
            setattr(self, key, item)


    def __repr__(self):
        string = "{}-Tensor(".format(self.dtype)
        if hasattr(self, "shape"):
            string += "shape={}, ".format(self.shape)
        if hasattr(self, "range"):
            string += "range={}, ".format(self.range)
        return string[:-2] + ")"


    def is_valid(self, value):
        # Check dtype:
        if self.dtype in ["cat", "N"]:
            if value.dtype != torch.int64:
                return False, "dtype"
        elif self.dtype == "real":
            if value.dtype != torch.float32:
                return False, "dtype"
        elif self.dtype == "bool":
            if value.dtype != torch.bool:
                return False, "dtype"
        # Check shape:
        if self.shape is not None and tuple(value.shape) != self.shape:
            return False, "shape"
        # Check range:
        if self.range is not None and (value.max() > max(self.range) or value.min() < min(self.range)):
            return False, "range"
        return True, None


# ### 1.1.2 Helper functions:

# In[ ]:


# Helper functions:
def get_SL_loss(
    graph_state,
    pred=None,
    w_op_dict=None,
    loss_type="mse",
    channel_coef=None,
    empty_coef=None,
    obj_coef=None,
    mutual_exclusive_coef=None,
    pixel_entropy_coef=None,
    pixel_gm_coef=None,
    iou_batch_consistency_coef=None,
    iou_concept_repel_coef=None,
    iou_relation_repel_coef=None,
    iou_relation_overlap_coef=None,
    iou_attract_coef=None,
    connected_coef=None,
    SGLD_is_anneal=None,
    SGLD_is_penalize_lower=None,
    SGLD_mutual_exclusive_coef=None,
    SGLD_pixel_entropy_coef=None,
    SGLD_pixel_gm_coef=None,
    SGLD_iou_batch_consistency_coef=None,
    SGLD_iou_concept_repel_coef=None,
    SGLD_iou_relation_repel_coef=None,
    SGLD_iou_relation_overlap_coef=None,
    SGLD_iou_attract_coef=None,
    lambd_start=None,
    lambd=None,
    image_value_range=None,
    w_init_type=None,
    indiv_sample=None,
    step_size=None,
    step_size_img=None,
    step_size_z=None,
    step_size_zgnn=None,
    step_size_wtarget=None,
    connected_num_samples=None,
    is_grad=True,
    isplot=0,
):
    """
    Get supervised learning loss on a graph_state.

    Args:
        graph_state: a GraphState instance
        loss_type: Choose from "mse", "ce" (cross-entropy), "l1".
        channel_coef: coeffient for the loss between predicted image and the target.
        empty_coef: coefficient for the empty channel
        obj_coef: coefficient for regularizing that each EBM discovers the objects in the target image
        mutual_exclusive_coef: coefficient for penalizing the overlap in mask.
        lambd_start: initial noise scale
        lambd: ending noise scale
        image_value_range: Minimum and maximum value for the values of the image at each pixel. For BabyARC/ARC, use "0,1", for CLEVR, use "-1,1".
        is_grad: whether allowing the gradient to flow through when computing the loss

    Returns:
        loss: a scalar of the loss.
    """
    g = graph_state.operator_graph
    selectors = g.get_selectors()
    if len(selectors) == 0:
        return None, None
    selector = selectors["Identity-1:Image"]
    input = graph_state.input[0][0]
    target = graph_state.target[0]
    loss, loss_dict = get_SL_loss_core(
        selector=selector,
        input=input,
        pred=pred,
        w_op_dict=w_op_dict,
        target=target,
        loss_type=loss_type,
        channel_coef=channel_coef,
        empty_coef=empty_coef,
        obj_coef=obj_coef,
        mutual_exclusive_coef=mutual_exclusive_coef,
        pixel_entropy_coef=pixel_entropy_coef,
        pixel_gm_coef=pixel_gm_coef,
        iou_batch_consistency_coef=iou_batch_consistency_coef,
        iou_concept_repel_coef=iou_concept_repel_coef,
        iou_relation_repel_coef=iou_relation_repel_coef,
        iou_relation_overlap_coef=iou_relation_overlap_coef,
        iou_attract_coef=iou_attract_coef,
        connected_coef=connected_coef,
        SGLD_is_anneal=SGLD_is_anneal,
        SGLD_is_penalize_lower=SGLD_is_penalize_lower,
        SGLD_mutual_exclusive_coef=SGLD_mutual_exclusive_coef,
        SGLD_pixel_entropy_coef=SGLD_pixel_entropy_coef,
        SGLD_pixel_gm_coef=SGLD_pixel_gm_coef,
        SGLD_iou_batch_consistency_coef=SGLD_iou_batch_consistency_coef,
        SGLD_iou_concept_repel_coef=SGLD_iou_concept_repel_coef,
        SGLD_iou_relation_repel_coef=SGLD_iou_relation_repel_coef,
        SGLD_iou_relation_overlap_coef=SGLD_iou_relation_overlap_coef,
        SGLD_iou_attract_coef=SGLD_iou_attract_coef,
        lambd_start=lambd_start,
        lambd=lambd,
        image_value_range=image_value_range,
        w_init_type=w_init_type,
        indiv_sample=indiv_sample,
        step_size=step_size,
        step_size_img=step_size_img,
        step_size_z=step_size_z,
        step_size_zgnn=step_size_zgnn,
        step_size_wtarget=step_size_wtarget,
        connected_num_samples=connected_num_samples,
        is_grad=is_grad,
        isplot=isplot,
    )
    return loss, loss_dict


def get_SL_loss_core(
    selector,
    input,
    pred,
    w_op_dict,
    target,
    loss_type,
    channel_coef,
    empty_coef,
    obj_coef,
    mutual_exclusive_coef,
    pixel_entropy_coef,
    pixel_gm_coef,
    iou_batch_consistency_coef,
    iou_concept_repel_coef,
    iou_relation_repel_coef,
    iou_relation_overlap_coef,
    iou_attract_coef,
    connected_coef,
    SGLD_is_anneal,
    SGLD_is_penalize_lower,
    SGLD_mutual_exclusive_coef,
    SGLD_pixel_entropy_coef,
    SGLD_pixel_gm_coef,
    SGLD_iou_batch_consistency_coef,
    SGLD_iou_concept_repel_coef,
    SGLD_iou_relation_repel_coef,
    SGLD_iou_relation_overlap_coef,
    SGLD_iou_attract_coef,
    lambd_start,
    lambd,
    image_value_range,
    w_init_type,
    indiv_sample,
    step_size,
    step_size_img,
    step_size_z,
    step_size_zgnn,
    step_size_wtarget,
    connected_num_samples,
    is_grad=True,
    isplot=0,
):
    """
    Get supervised learning loss on a selector.

    Args (priority: the arguments passed in > selector's attributes > default values):
        selector: a selector
        input: a tensor with shape [B, C:10, H, W]
        target: a tensor with shape [B, C:10, H, W]
        loss_type: Choose from "mse", "ce" (cross-entropy), "l1".
        channel_coef: coeffient for the loss between predicted image and the target.
        empty_coef: coefficient for the empty channel.
        obj_coef: coefficient for regularizing that each EBM discovers the objects in the target image.
        mutual_exclusive_coef: coefficient for penalizing the overlap in mask.
        lambd_start: initial noise scale
        lambd: ending noise scale
        image_value_range: Minimum and maximum value for the values of the image at each pixel. For BabyARC/ARC, use "0,1", for CLEVR, use "-1,1".
        is_grad: whether allowing the gradient to flow through when computing the loss

    Returns:
        loss: a scalar of the loss.
    """
    # is_grad=True is important to be able to pass gradient back to the SGLD:
    if w_op_dict is None:
        assert pred is None
        device = input.device
        pred, w_op_dict = selector.forward_NN(
            input,
            is_grad=is_grad,
            lambd_start=lambd_start,
            lambd=lambd,
            SGLD_is_anneal=SGLD_is_anneal,
            SGLD_is_penalize_lower=SGLD_is_penalize_lower,
            SGLD_mutual_exclusive_coef=SGLD_mutual_exclusive_coef,
            SGLD_pixel_entropy_coef=SGLD_pixel_entropy_coef,
            SGLD_pixel_gm_coef=SGLD_pixel_gm_coef,
            SGLD_iou_batch_consistency_coef=SGLD_iou_batch_consistency_coef,
            SGLD_iou_concept_repel_coef=SGLD_iou_concept_repel_coef,
            SGLD_iou_relation_repel_coef=SGLD_iou_relation_repel_coef,
            SGLD_iou_relation_overlap_coef=SGLD_iou_relation_overlap_coef,
            SGLD_iou_attract_coef=SGLD_iou_attract_coef,
            image_value_range=image_value_range,
            w_init_type=w_init_type,
            indiv_sample=indiv_sample,
            step_size=step_size,
            step_size_img=step_size_img,
            step_size_z=step_size_z,
            step_size_zgnn=step_size_zgnn,
            step_size_wtarget=step_size_wtarget,
            isplot=isplot,
        )
    else:
        w_0 = w_op_dict[next(iter(w_op_dict))]
        device = w_0.device
    if len(w_0.shape) == 5:
        batch_shape = tuple(target.shape[:2])  # [B_task, B_example]
        target = target.view(-1, *target.shape[-3:])
        if pred is not None:
            pred = pred.view(-1, *pred.shape[-3:])
        w_op_dict = {key: item.view(-1, *item.shape[-3:]) for key, item in w_op_dict.items()}
    else:
        batch_shape = None

    assert len(target.shape) == 4
    assert pred is None or len(pred.shape) == 4
    assert len(w_op_dict[next(iter(w_op_dict))].shape) == 4

    channel_coef = channel_coef if channel_coef is not None else selector.channel_coef if selector.channel_coef is not None else 1
    obj_coef = obj_coef if obj_coef is not None else selector.obj_coef if selector.obj_coef is not None else 0.1
    empty_coef = empty_coef if empty_coef is not None else selector.empty_coef if selector.empty_coef is not None else 0.02
    mutual_exclusive_coef = mutual_exclusive_coef if mutual_exclusive_coef is not None else selector.mutual_exclusive_coef if selector.mutual_exclusive_coef is not None else 0.1
    pixel_entropy_coef = pixel_entropy_coef if pixel_entropy_coef is not None else selector.pixel_entropy_coef if selector.pixel_entropy_coef is not None else 0.
    pixel_gm_coef = pixel_gm_coef if pixel_gm_coef is not None else selector.pixel_gm_coef if selector.pixel_gm_coef is not None else 0.
    iou_batch_consistency_coef = iou_batch_consistency_coef if iou_batch_consistency_coef is not None else selector.iou_batch_consistency_coef if selector.iou_batch_consistency_coef is not None else 0.
    iou_concept_repel_coef = iou_concept_repel_coef if iou_concept_repel_coef is not None else selector.iou_concept_repel_coef if selector.iou_concept_repel_coef is not None else 0.
    iou_relation_repel_coef = iou_relation_repel_coef if iou_relation_repel_coef is not None else selector.iou_relation_repel_coef if selector.iou_relation_repel_coef is not None else 0.
    iou_relation_overlap_coef = iou_relation_overlap_coef if iou_relation_overlap_coef is not None else selector.iou_relation_overlap_coef if selector.iou_relation_overlap_coef is not None else 0.
    iou_attract_coef = iou_attract_coef if iou_attract_coef is not None else selector.iou_attract_coef if selector.iou_attract_coef is not None else 0.

    loss_dict = {}
    loss = torch.tensor(0., dtype=torch.float32).to(device)
    if obj_coef > 0 and pred is not None:
        loss_obj = get_obj_loss(pred, w_op_dict, target, loss_type=loss_type) * obj_coef
        loss = loss + loss_obj
        loss_dict["loss_obj"] = to_np_array(loss_obj)

    if loss_type == "mse":
        loss_fun = nn.MSELoss()
    elif loss_type == "l1":
        loss_fun = nn.L1Loss()

    if pred is not None:
        if pred.shape[1] == 10:
            if loss_type in ["mse", "l1"]:
                if channel_coef > 0:
                    loss_channel = loss_fun(pred[:,1:], target[:,1:]) * channel_coef
                else:
                    loss_channel = torch.tensor(0., dtype=torch.float32).to(device)
                if empty_coef > 0:
                    loss_empty = loss_fun(pred[:,:1], target[:,:1]) * empty_coef
                else:
                    loss_empty = torch.tensor(0., dtype=torch.float32).to(device)
            else:
                raise Exception("loss_type {} is not valid!".format(loss_type))
        else:
            # No channel dedicated to background pixels
            assert pred.shape[1] == 3 or pred.shape[1] == 2
            if loss_type in ["mse", "l1"]:
                if channel_coef > 0:
                    loss_channel = loss_fun(pred, target) * channel_coef
                else:
                    loss_channel = torch.tensor(0., dtype=torch.float32).to(device)
                loss_empty = torch.tensor(0., dtype=torch.float32).to(device)
            else:
                raise
    else:
        loss_channel = torch.tensor(0., dtype=torch.float32).to(device)
        loss_empty = torch.tensor(0., dtype=torch.float32).to(device)

    loss = loss + loss_channel + loss_empty
    loss_dict["loss_channel"] = to_np_array(loss_channel)
    loss_dict["loss_empty"] = to_np_array(loss_empty)

    mask_list = list(w_op_dict.values())
    w_op = mask_list[0]

    # Mutual exclusive loss for mask:
    if mutual_exclusive_coef > 0 and w_op.shape[1] == 1:
        loss_mask_overlap = get_neg_mask_overlap(mask_list, mask_info=selector.get_mask_info()).mean() * mutual_exclusive_coef
        loss = loss + loss_mask_overlap
        loss_dict["loss_mask_overlap"] = to_np_array(loss_mask_overlap)

    if pixel_entropy_coef > 0 and w_op.shape[1] == 1:
        loss_pixel_entropy = get_pixel_entropy(mask_list).mean() * pixel_entropy_coef
        loss = loss + loss_pixel_entropy
        loss_dict["loss_pixel_entropy"] = to_np_array(loss_pixel_entropy)

    if pixel_gm_coef > 0 and w_op.shape[1] == 1:
        loss_pixel_gm = get_pixel_gm(mask_list).mean() * pixel_gm_coef
        loss = loss + loss_pixel_gm
        loss_dict["loss_pixel_gm"] = to_np_array(loss_pixel_gm)
        
    if connected_coef > 0 and w_op.shape[1] == 1:
        loss_connected = get_connected_loss(torch.cat(mask_list), connected_num_samples).mean() * connected_coef
        loss = loss + loss_connected
        loss_dict["loss_connected"] = to_np_array(loss_connected)

    if w_op.shape[1] == 1 and (
        iou_batch_consistency_coef > 0 or
        iou_concept_repel_coef > 0 or
        iou_relation_repel_coef > 0 or
        iou_relation_overlap_coef > 0 or
        iou_attract_coef > 0
    ):
        loss_iou, loss_iou_dict = get_graph_energy(
            mask_list,
            mask_info=selector.get_mask_info(),
            iou_batch_consistency_coef=iou_batch_consistency_coef,
            iou_concept_repel_coef=iou_concept_repel_coef,
            iou_relation_repel_coef=iou_relation_repel_coef,
            iou_relation_overlap_coef=iou_relation_overlap_coef,
            iou_attract_coef=iou_attract_coef,
            batch_shape=batch_shape,
        )
        loss = loss + loss_iou.mean()
        loss_iou_dict_mean = {key: item.mean() for key, item in loss_iou_dict.items()}
        loss_dict.update(loss_iou_dict_mean)

    return loss, loss_dict


def get_obj_loss(pred, w_op_dict, target, loss_type="mse"):
    """Get loss on individual discovered objects.

    4 scenarios:
        (1) pred.shape[1] == 10 and w_op.shape[1] == 1:  BabyARC, each w_op is a mask from 1st SGLD, and pred comes from 2nd SGLD
        (2) pred.shape[1] == 10 and w_op.shape[1] == 10: BabyARC, each w_op is an object from 1st SGLD, and pred comes from combining objs from w_op_dict
        (3) pred.shape[1] == 3  and w_op.shape[1] == 1:  CLEVR, each w_op is a mask from 1st SGLD, and pred comes from 2nd SGLD
        (4) pred.shape[1] == 3  and w_op.shape[1] == 3:  CLEVR, each w_op is an object from 1st SGLD. In this case no object loss, and pred comes from combining objs from w_op_dict
    """
    if loss_type == "mse":
        loss_fun = nn.MSELoss()
    elif loss_type == "l1":
        loss_fun = nn.L1Loss()
    else:
        raise Exception("loss_type {} is not valid!".format(loss_type))

    loss_obj = torch.tensor(0., dtype=torch.float32).to(target.device)
    for key, w_op in w_op_dict.items():
        if pred.shape[1] == 10:
            if w_op.shape[1] == 1:
                # w_op is a mask:
                loss_obj_ele = loss_fun(pred*w_op, target*w_op)
            else:
                assert w_op.shape[1] == 10
                mask = (w_op.argmax(1) != 0)[:, None]
                loss_obj_ele = loss_fun(w_op*mask, target*mask)
        else:
            assert pred.shape[1] == 3 or pred.shape[1] == 2
            if w_op.shape[1] == 1:
                loss_obj_ele = loss_fun(pred*w_op, target*w_op)
            else:
                loss_obj_ele = 0
        loss_obj = loss_obj + loss_obj_ele
    return loss_obj


def copy_helper(to_copy_dict, is_share_fun=False, is_copy_module=True, global_attrs=None):
    """
    Deepcopy a dictionary, taking into consideration about torch.nn.Modules and Tensors with grads.

    Args:
        is_copy_module: if True, will copy torch.nn.Module. Otherwise not copy.
        global_attrs: a list of class attribute names that are global dictionaries.

    Returns:
        copied_dict: The deepcopied dictionary.
        global_dicts: the global dictionaries as class attributes, if the arg global_attrs is not None.
    """
    global_dicts = {}
    if isinstance(to_copy_dict, torch.Tensor):
        if to_copy_dict.requires_grad:
            # Copy tensor, detach it from the original computation graph, then allow gradients again
            to_copy_dict = to_copy_dict.clone().detach().requires_grad_()
        else:
            to_copy_dict = to_copy_dict.clone()
        return to_copy_dict, global_dicts

    copied_dict = {}
    if global_attrs is None:
        global_attrs = []
    if not isinstance(global_attrs, list):
        global_attrs = [global_attrs]
    if isinstance(to_copy_dict, OrderedDict):
        copied_dict = OrderedDict()
    for name, value in to_copy_dict.items():
        if isinstance(value, Placeholder):
            copied_dict[name] = value.copy_with_grad(is_copy_module=is_copy_module, global_attrs=global_attrs)
        elif isinstance(value, dict):
            if name in global_attrs:
                global_dicts[name] = value
            else:
                copied_dict[name], global_dicts_ele = copy_helper(value, is_copy_module=is_copy_module, global_attrs=global_attrs)
        elif isinstance(value, nn.Module) and not isinstance(value, nx.Graph):
            if is_copy_module:
                other_attr = []
                if value.__class__.__name__ == "ConceptEBM":
                    other_attr += ["c_repr", "c_str"]
                copied_dict[name] = copy_with_model_dict(value, other_attr=other_attr)
        elif isinstance(value, torch.Tensor):
            if value.requires_grad:
                # Copy tensor, detach it from the original computation graph, then allow gradients again
                copied_dict[name] = value.clone().detach().requires_grad_()
            else:
                copied_dict[name] = value.clone()
        elif isinstance(value, BaseGraph):
            copied_dict[name] = value.copy_with_grad(is_share_fun=is_share_fun, is_copy_module=is_copy_module, global_attrs=global_attrs)
        elif isinstance(value, tuple):
            copied_dict[name] = tuple(copy_helper(element, is_share_fun=is_share_fun, is_copy_module=is_copy_module, global_attrs=global_attrs)[0] for element in value)
        elif not isinstance(value, NodeView):
            try:
                copied_dict[name] = deepcopy(value)
            except Exception as e:
                pdb.set_trace()
    return copied_dict, global_dicts


def init_tensor(placeholder, is_cuda=False):
    """Initialize PyTorch tensor according to the specs of the placeholder.

    placeholder.mode: "Pos", "RelPos", "Cat", "Bool"
    """
    concept = CONCEPTS[placeholder.mode]
    assert len(concept.nodes) == 1
    tensor_spec = concept.get_node_content(placeholder.mode).mode
    assert isinstance(tensor_spec, Tensor)
    drange = tensor_spec.range if tensor_spec.range is not None else placeholder.range
    dshape = tensor_spec.shape if tensor_spec.shape is not None else placeholder.shape
    tensor = to_Variable(np.random.rand(*dshape) * (max(drange) - min(drange)) + min(drange), is_cuda=is_cuda)
    return tensor


def check_input_valid(op, *obj_names):
    """Returns True if the obj_names (e.g. [obj_1:Image, obj_2:Line]) is valid for the op (considering concept inheritance)."""
    is_valid = True
    for i, obj_name in enumerate(obj_names):
        input_placeholder = Placeholder(op.input_placeholder_nodes[i].split(":")[-1])
        obj_placeholder = Placeholder(obj_name.split(":")[-1])
        is_valid = input_placeholder.accepts(obj_placeholder)
        if not is_valid:
            is_valid = False
            break
    return is_valid


class MyBounds(object):
    """Used to bound the basinhopping search function implemented below in search()"""
    def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
        
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def init_ebm_dict(
    modes,
    ebm_mode,
    CONCEPTS=None,
    OPERATORS=None,
    cache_forward=True,
    device="cpu",
    **kwargs
):
    """Initialize the ebm_dict with the given modes."""
    num_colors = 10
    selector = Concept_Pattern(
        name=None,
        value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
        attr={},
        is_all_obj=True,
        is_ebm=True,
        is_default_ebm=False,
        ebm_dict={},
        CONCEPTS=CONCEPTS,
        OPERATORS=OPERATORS,
        device=device,
        cache_forward=cache_forward,
        **kwargs
    )
    ebm_dict = selector.ebm_dict
    assert len(ebm_dict) == 0
    for mode in modes:
        selector.init_ebm(
            method="random",
            mode=mode,
            ebm_mode=ebm_mode,
            ebm_model_type="CEBM",
            CONCEPTS=CONCEPTS,
            OPERATORS=OPERATORS,
        )
    return ebm_dict


# ## 1.2 BaseGraph:

# In[ ]:


class BaseGraph(nx.MultiDiGraph, nn.Module):
    def __init__(self, G=None, **kwargs):
        """Backbone of Concept() and Graph() classes:

        Contains basic methods for manipulating the graph, and visualization.
        """
        self.is_cuda = kwargs["is_cuda"] if "is_cuda" in kwargs else False
        self.verbose = kwargs["verbose"] if "verbose" in kwargs else True
        if G is not None and G.device is not None:
            self.device = G.device
        else:
            self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")
        BaseGraph.__mro__[-2].__init__(self)  # Obtain the superclass nn.Module
        if "name" in kwargs:
            super(BaseGraph, self).__init__(incoming_graph_data=G, name=kwargs["name"])
        else:
            super(BaseGraph, self).__init__(incoming_graph_data=G)


    def copy_with_grad(self, is_share_fun=False, is_copy_module=True, global_attrs=None):
        """Return the copy of current instance by detaching tensors which have grad
        and deepcopying all other object attributes.

        Args:
            is_share_fun: if True, the copy will share its torch.nn.Modules with its original.
            is_copy_module: if True, will copy torch.nn.Module. Otherwise not copy.
            global_attrs: a list of class attribute names that are global dictionaries.

        Returns:
            G: the copied class instance.
        """
        G_copy = self.__class__()
        copied_dict, global_dicts = copy_helper(self.__dict__, is_copy_module=is_copy_module, global_attrs=global_attrs)
        G_copy.__dict__.update(copied_dict)
        G = self.__class__(G=G_copy)

        if global_attrs is not None:
            # Set the global dictionaries as attributes of the class:
            for key, Dict in global_dicts.items():
                setattr(G, key, Dict)
        if is_share_fun:
            for fun_name in self.funs:
                setattr(G, "fun_" + fun_name, getattr(self, "fun_" + fun_name))
                G.set_node_content(getattr(G, "fun_" + fun_name), fun_name)
        return G


    def copy(self, is_share_fun=False):
        """Return the copy of current instance. Warning: does not work when passing
        gradients through graph."""
        G = deepcopy(self)
        if is_share_fun:
            for fun_name in self.funs:
                setattr(G, "fun_" + fun_name, getattr(self, "fun_" + fun_name))
                G.set_node_content(getattr(G, "fun_" + fun_name), fun_name)
        return G


    def copy_shallow(self):
        """Return a shallow copy of current instance."""
        return super(nx.MultiDiGraph, self).copy()


    def to(self, device):
        super(BaseGraph, self).to(device)
        self.device = device
        return self


    ########################################
    # Obtain or modify node content and neighbors:
    ########################################
    def get_node_name(self, node_name=None):
        """Get full node_name. If node_name is None (assuming only one node in the graph), 
        return the unique node_name."""
        if node_name is None:
            node_name = self.name
            if self.name is None:
                if len(self.nodes) == 1:
                    node_name = list(self.nodes)[0]
        elif node_name == "$root":
            node_name = self.name
        else:
            if ":" in node_name:
                pass
            else:
                if node_name in self.nodes:
                    pass
                else:
                    is_exist = False
                    for node in sorted(self.nodes):
                        if node.startswith(node_name + ":"):
                            node_name = node
                            is_exist = True
                            break
                    assert is_exist, "node '{}' does not exist!".format(node_name)
        return node_name


    def get_node_content(self, node_name=None):
        """Obtain the content of a node. The content can be a placeholder, a Concept(), or a Graph()."""
        node_name = self.get_node_name(node_name)
        if "value" in self.nodes(data=True)[node_name]:
            return self.nodes(data=True)[node_name]["value"]
        else:
            return None


    def set_node_content(self, content, node_name=None):
        """Set the content of a node. The content can be a placeholder, a Concept(), or a Graph()."""
        node_name = self.get_node_name(node_name)
        self.nodes(data=True)[node_name]["value"] = content
        return self


    def get_node_type(self, node_name=None):
        """Obtain the type of a node."""
        node_name = self.get_node_name(node_name)
        return self.nodes(data=True)[node_name]["type"]


    def get_node_repr(self, node_name=None):
        """Obtain the repr (embedding) of a concept node."""
        node_name = self.get_node_name(node_name)
        if "repr" in self.nodes(data=True)[node_name]:
            return self.nodes(data=True)[node_name]["repr"]
        else:
            return None


    def operator_name(self, operator):
        """Get the name of the operator. If it is an attribute node, preserve the attribute."""
        if "^" in operator or "input" in operator or "concept" in operator:
            # If it is an attribute node:
            return operator.split(":")[0]
        else:
            if "-" in operator:
                return operator.split("-")[0]
            else:
                return operator.split(":")[0]


    def get_node_fun(self, node_name=None):
        """Obtain the fun of a concept node."""
        node_name = self.get_node_name(node_name)
        if "fun" in self.nodes(data=True)[node_name]:
            return self.nodes(data=True)[node_name]["fun"]
        else:
            return None


    def get_node_value(self, node_name=None):
        """Get the value held in the Placeholder of a node, if the content is a Placeholder."""
        if not isinstance(self.get_node_content(node_name), Placeholder):
            return None
        else:
            node_name = self.get_node_name(node_name)
            if "fun" in self.nodes(data=True)[node_name] and self.nodes(data=True)[node_name]["fun"] is not None and len(self.parent_nodes(node_name)) > 0:
                fun = self.nodes(data=True)[node_name]["fun"]
                value = fun(self.get_node_value(self.parent_nodes(node_name)[0]))
                self.nodes(data=True)[node_name]["value"].value = value
                return value
            else:
                return self.nodes(data=True)[node_name]["value"].value


    def set_node_value(self, value, node_name=None):
        """Set up the value held in the Placeholder of a node, if the content is a Placeholder."""
        assert isinstance(self.get_node_content(node_name), Placeholder)
        node_name = self.get_node_name(node_name)
        if not isinstance(value, torch.Tensor):
            self.nodes(data=True)[node_name]["value"].value = to_Variable(value, is_cuda=self.is_cuda)
        else:
            self.nodes(data=True)[node_name]["value"].value = value
        return self


    def remove_node_value(self, node_name=None):
        """Delete the value held in the Placeholder of a node, if the content is a Placeholder."""
        assert isinstance(self.get_node_content(node_name), Placeholder)
        node_name = self.get_node_name(node_name)
        self.nodes(data=True)[node_name]["value"].value = None
        return self


    def parent_nodes(self, node_name):
        """Obtain the parent nodes of a given node."""
        parent_nodes = []
        node_name = self.get_node_name(node_name)
        for node, adj in self[node_name].items():
            if adj[0]["type"].startswith("b-") and "relation" not in adj[0]["type"]:
                parent_nodes.append(node)
        return parent_nodes


    def child_nodes(self, node_name):
        """Obtain the child nodes of a given node."""
        child_nodes = []
        node_name = self.get_node_name(node_name)
        for node, adj in self[node_name].items():
            if not adj[0]["type"].startswith("b-") and "relation" not in adj[0]["type"]:
                child_nodes.append(node)
        return child_nodes

    
    def get_edge_type(self, node1, node2):
        """Get the type of the edge from node1 to node2."""
        node1 = self.get_node_name(node1)
        node2 = self.get_node_name(node2)
        return self.edges[(node1, node2, 0)]["type"]


    def set_edge_type(self, node1, node2, type):
        """Set the type of the edge from node1 to node2."""
        node1 = self.get_node_name(node1)
        node2 = self.get_node_name(node2)
        self.edges[(node1, node2, 0)]["type"] = type
        self.edges[(node2, node1, 0)]["type"] = "b-{}".format(type)
        return self


    def get_attr_source(self, attr_name):
        """Get the source node of an attribute (either an input node, concept node or fun-out node)"""
        attr_name = self.get_node_name(attr_name)
        assert self.get_node_type(attr_name) == "attr"
        parent_node = attr_name
        while self.get_node_type(parent_node) not in ["input", "fun-out", "concept"]:
            parent_node = self.parent_nodes(parent_node)
            assert len(parent_node) == 1
            parent_node = parent_node[0]
        return parent_node


    def get_path(self, source_node, target_node):
        """Get a path from the source_node to the target_node."""
        source_node = self.get_node_name(source_node)
        target_node = self.get_node_name(target_node)
        try:
            path = nx.shortest_path(self, source_node, target_node)
            is_connected = True
        except:
            path = []
            is_connected = False
        return is_connected, path


    def get_path_to_output(self, node_name):
        """Get the path from the current operator to its output."""
        node_name = self.get_node_name(node_name)
        path = [node_name]
        while len(self.child_nodes(node_name)) > 0:
            node_name = self.child_nodes(node_name)
            if len(node_name) == 0:
                return path
            if "inter" in self[path[-1]][node_name[0]][0]["type"]:
                return path
            if len(node_name) > 1:
                print("The node {} has more than 1 output. Choose {}.".format(path[-1], node_name[0]))
                path.append(node_name[0])
            else:
                path.append(node_name[0])
            node_name = node_name[0]
        return path


    def get_to_outnode(self, node_name):
        """Get the path from the current operator to its output."""
        node_name = self.get_node_name(node_name)
        if self.get_node_type(node_name) == "fun-out":
            return node_name
        while len(self.child_nodes(node_name)) > 0:
            node_name_new = self.child_nodes(node_name)
            if len(node_name_new) == 0:
                return node_name
            if 'intra-attr' in self[node_name][node_name_new[0]][0]["type"] or "inter" in self[node_name][node_name_new[0]][0]["type"]:
                return node_name
            node_name = node_name_new[0]
        return node_name


    def get_node_neighboring_inter(self, node):
        """Get the node within the operator or input_placeholder that connect to an inter-edge."""
        node = self.get_node_name(node)
        child_nodes = self.child_nodes(node)
        if len(child_nodes) == 0:
            return None
        for adj, edge in self[node].items():
            if "inter" in edge[0]["type"]:
                return node
            else:
                if not edge[0]["type"].startswith("b-"):
                    node_cand = self.get_node_neighboring_inter(adj)
                    if node_cand is not None:
                        return node_cand
        return None


    def get_ancestors(self, nodes, includes_self=False):
        """Get all the ancestors of the nodes given."""
        if not isinstance(nodes, list):
            nodes = [nodes]
        ancestors = []
        forward_graph = self.core_forward_graph_shallow
        for node in nodes:
            node = self.get_node_name(node)
            ancestors += nx.ancestors(forward_graph, node)
            if includes_self:
                ancestors.append(node)
        return remove_duplicates(ancestors)


    def get_descendants(self, nodes, includes_self=False):
        """Get all the descendants of the nodes given."""
        if not isinstance(nodes, list):
            nodes = [nodes]
        descendants = []
        forward_graph = self.core_forward_graph_shallow
        for node in nodes:
            node = self.get_node_name(node)
            descendants += nx.descendants(forward_graph, node)
            if includes_self:
                descendants.append(node)
        return remove_duplicates(descendants)


    def check_available(self, node_name):
        """Check if any the current node are connected to an inter-edge."""
        node_name = self.get_node_name(node_name)
        assert self.get_node_type(node_name) in ["attr", "input", "fun-out"]
        is_available = True
        for neighbor_node, info in self[node_name].items():
            if "inter" in info[0]["type"]:
                is_available = False
                break
        return is_available


    def check_available_recur(self, node_name):
        """Check if any the current node and its descendants are connected to an inter-edge."""
        if not self.check_available(node_name):
            return False
        for child_node in self.child_nodes(node_name):
            if not self.check_available_recur(child_node):
                return False
        return True


    def check_dangling(self, node_name):
        """Check if any of the current node, its ancestors and descendants are connected to an inter-edge."""
        node_name = self.get_node_name(node_name)
        assert self.get_node_type(node_name) in ["attr", "input", "fun-out"]
        if not self.check_available_recur(node_name):
            return False
        # Check all its ancestors:
        if self.get_node_type(node_name) == "attr":
            parent_name = node_name
            while self.get_node_type(parent_name) in ["attr", "input", "fun-out"]:
                if not self.check_available(parent_name):
                    return False
                if self.get_node_type(parent_name) in ["input", "fun-out"]:
                    break
                parent_name = self.parent_nodes(parent_name)
                assert len(parent_name) == 1
                parent_name = parent_name[0]
        return True


    ########################################
    # Properties:
    ########################################
    def forward_graph(self, is_copy_module=True, global_attrs=None):
        """Get the forward graph."""
        if is_copy_module is False or global_attrs is not None:
            G = self.copy_with_grad(is_copy_module=is_copy_module, global_attrs=global_attrs)
        else:
            G = self.copy()
        backward_edges = []
        for ni, no, data in G.edges(data=True):
            if data["type"].startswith("b-"):
                backward_edges.append((ni, no))
        G.remove_edges_from(backward_edges)
        return G


    @property
    def core_graph(self):
        """Get the graph without relation edges"""
        G = self.copy()
        relation_edges = []
        for ni, no, data in G.edges(data=True):
            if "relation" in data["type"]:
                relation_edges.append((ni, no))
        G.remove_edges_from(relation_edges)
        return G


    @property
    def core_forward_graph_shallow(self):
        G = self.copy_shallow()
        relation_edges = []
        for ni, no, data in G.edges(data=True):
            if "relation" in data["type"] or data["type"].startswith("b-"):
                relation_edges.append((ni, no))
        G.remove_edges_from(relation_edges)
        return G


    @property
    def obj_graph(self):
        """Obtain the graph only containing object nodes."""
        G = self.get_subgraph(self.obj_names, includes_root=False, includes_descendants=False)
        return G


    def get_graph(self, allowed_attr):
        """Get subgraph based on allowed_attr: all (includes all nodes) or obj (includes only object nodes)."""
        if allowed_attr == "all":
            return self
        elif allowed_attr == "obj":
            return self.obj_graph
        else:
            raise Exception("allowed_attr '{}' is not valid!".format(allowed_attr))


    @property
    def backward_graph(self):
        """Get the backward graph."""
        G = self.copy()
        forward_edges = []
        for ni, no, data in G.edges(data=True):
            if not data["type"].startswith("b-"):
                forward_edges.append((ni, no))
        G.remove_edges_from(forward_edges)
        return G


    def clean_graph(self, is_copy_module=True):
        """Draw a clean operator graph where the operaoters' input and output nodes connected
        by an inter-edge is absorbed.
        """
        try:
            G = self.copy_with_grad(is_copy_module=is_copy_module)
        except:
            assert isinstance(self, Concept)
            return self
        nodes_to_remove = []
        edges_to_remove = []
        for node1, node2, edge in G.forward_graph(is_copy_module=is_copy_module).edges(data=True):
            if edge["type"].startswith("inter"):
                if G.get_node_type(node2) == "fun-in":
                    node2_parents = G.parent_nodes(node2)
                    node2_child = G.child_nodes(node2)[0]
                    assert G.get_node_type(node2_child) == "self"
                    for node2_parent in node2_parents:
                        edges_to_remove.append((node2_parent, node2))
                        edges_to_remove.append((node2, node2_parent))
                        G.add_edge(node2_parent, node2_child, type=self.edges[(node2_parent, node2, 0)]["type"])
                    nodes_to_remove.append(node2)

                if G.get_node_type(node1) == "fun-out":
                    node1_children = G.child_nodes(node1)
                    node1_parent = G.parent_nodes(node1)
                    if len(node1_parent) > 0:
                        node1_parent = node1_parent[0]
                        if G.get_node_type(node1_parent) == "self":
                            for node1_child in node1_children:
                                edges_to_remove.append((node1, node1_child))
                                edges_to_remove.append((node1_child, node1))
                                G.add_edge(node1_parent, node1_child,
                                           type=self.edges[(node1, node1_child, 0)]["type"] if (node1, node1_child, 0) in self.edges else edge["type"])
                            nodes_to_remove.append(node1)
            elif edge["type"] == "intra-attr" and G.get_node_type(node1) == "fun-out":
                # If it is an get-attr node from a fun-out node:
                node1_children = G.child_nodes(node1)
                node1_parents = G.parent_nodes(node1)
                assert len(node1_parents) == 1
                node1_parent = node1_parents[0]
                for node1_child in node1_children:
                    edges_to_remove.append((node1, node1_child))
                    edges_to_remove.append((node1_child, node1))
                    G.add_edge(node1_parent, node1_child,
                               type=self.edges[(node1, node1_child, 0)]["type"] if (node1, node1_child, 0) in self.edges else edge["type"])
                nodes_to_remove.append(node1)

        G.remove_nodes_from(nodes_to_remove)
        G.remove_edges_from(edges_to_remove)
        G = G.forward_graph(is_copy_module=is_copy_module) if not nx.is_directed_acyclic_graph(G) else G
        return G


    @property
    def topological_sort(self):
        """Return a list of nodes that are topologically sorted with the full graph."""
        return list(nx.lexicographical_topological_sort(self.core_forward_graph_shallow))


    @property
    def node_link_data(self):
        """Obtain node and link data for web app."""
        G = self.copy().forward_graph()
        for n in G:
            G.nodes[n]['id'] = n
        data = {"graph_type": G.__class__.__name__}
        data.update(json_graph.node_link_data(G))
        node_reverse_dict = {}

        # Nodes:
        for i, node in enumerate(data["nodes"]):
            node.pop("repr", None)
            node.pop("fun", None)
            node.pop("value",None)
            node_reverse_dict[node["id"]] = i
#             if "value" in node:
#                 if not isinstance(node["value"], BaseGraph):
#                     node.pop("value", None)
#                 else:
#                     node["value"] = node["value"].node_link_data

        # Links:
        for link in data["links"]:
            link["source"] = node_reverse_dict[link["source"]]
            link["target"] = node_reverse_dict[link["target"]]
        
        # Remove other redundent information:
        data.pop("graph", None)
        data.pop("directed", None)
        data.pop("multigraph", None)
        return data


    ########################################
    # Visualization:
    ########################################
    def draw(self, is_clean_graph=True, layout="spring", filename=None, **kwargs):
        """Visualize the current graph."""
        # Only plot the acyclic edges, and by default the forward graph:
        if is_clean_graph:
            G = self.clean_graph(is_copy_module=False)
        else:
            G = self.forward_graph(is_copy_module=False) if not nx.is_directed_acyclic_graph(self) else self
        # Nodes:
        node_color = []
        node_sizes = []
        for node, info in G.nodes(data=True):
            node_size = 1200
            if info["type"] == "input":
                node_color.append("#58509d")
            elif info["type"] == "concept":
                node_color.append("#C515E8")
                if self.get_node_value(node) is not None:
                    node_size = 1800
            elif info["type"] == "self":
                node_color.append("#1f78b4")
            elif info["type"] == "attr":
                node_color.append("#EB8F8F")
            elif info["type"] == "obj":
                node_color.append("#C31B37")
            elif info["type"].startswith("fun"):
                if info["type"] == "fun-out":
                    node_color.append("orange")
                else:
                    node_color.append("g")
            else:
                raise
            node_sizes.append(node_size)

        # Edges:
        edge_color = []
        for ni, no, data in G.edges(data=True):
            if "intra-relation" in data["type"]:
                edge_color.append("purple")
            elif "inter-concept" in data["type"]:
                edge_color.append("#E644F0")
            elif "intra" in data["type"]:
                edge_color.append("k")
            elif "inter" in data["type"]:
                if data["type"].endswith("inter-input"):
                    edge_color.append("brown")
                elif data["type"].endswith("inter-criteria"):
                    edge_color.append("c")
                else:
                    raise
            elif data["type"].endswith("get-attr"):
                    edge_color.append("#E815DA")
            else:
                raise
        # Set up layout:
        if layout == "planar":
            pos = nx.planar_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        elif layout == "spiral":
            pos = nx.spiral_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G)
        elif layout == "kk":
            pos = nx.kamada_kawai_layout(G)
        else:
            raise

        # Draw:
        nx.draw(G, with_labels=True, font_size=10, pos=pos,
                node_color=node_color, node_size=node_sizes,
                edge_color=edge_color,
               )
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", **kwargs)
        else:
            plt.show()

            if isinstance(self, Concept) and self.name is not None:
                value = self.get_node_value(self.name)
                if value is not None:
                    if len(value.shape) == 2:
                        visualize_matrices([value])
                    elif len(value.shape) == 3:
                        value_T = deepcopy(value).permute(1,2,0)
                        plt.imshow(to_np_array(value_T).astype(int))
                        plt.show()

            # If certain node's content is a graph, recursively draw it:
            for node in self.nodes:
                node_content = self.get_node_content(node)
                if isinstance(node_content, BaseGraph):
                    print("\nDrawing the content of node '{}', which is {}:".format(node, node_content))
                    node_content.draw()

            # Draw selectors:
            if isinstance(self, Graph):
                for op_name in self.operators:
                    selector_dict = self.get_selector(op_name)
                    if len(selector_dict) > 0:
                        for key, selector in selector_dict.items():
                            print("\nOp-in '{}' has a selector, with inplace={}:".format(key, self.get_inplace(key)))
                            selector.draw()
        return self


    def visualize_results(self, results):
        for node, value in results.items():
            if len(value.shape) == 2:
                print("{}:".format(node))
                visualize_matrices([value])
            elif len(value.shape) == 3:
                print("{}:".format(node))
                visualize_matrices(value)
            else:
                print("{}: {}".format(node, value))


    def save(self, filename):
        """Save the graph instance to a file."""
        pickle.dump(self, open(filename, "wb"))
        return self


def load_graph(filename):
    """Load the graph instance."""
    return pickle.load(open(filename, "rb"))


# ## 1.3 Graph:

# ### 1.3.1 Graph

# In[ ]:


class Graph(BaseGraph):
    """Implement the Graph class where operators, input and programs are its special cases.

    A operator is a node with (optional) attributes and (optional) methods. 
        Attributes and input, output of the methods can be other operators, 
        and are modeled as nodes.
    An input is a Graph with nodes.
    A program is a directed acyclic graph (DAG) with overall input nodes and output nodes. 
        The input nodes can be grounded (connected) to the input graph, and the output 
        node(s) generates the output of the program. In case where some of the intermediate 
        input nodes are not grounded, the forward mode of the program will generate multiple 
        outputs corresponding to the combined ranges of the ungrounded input nodes.
    It is inherited from MultiDiGraph for graph manipulation and torch.nn.Module for 
        gradient-based learning.
    """
    def __init__(self, G=None, **kwargs):
        """Components of a Graph instance:

        (1) Nodes and edges: a node can be a operator body node, an attribute node for a operator, 
            an input or output node of a operator. Also there is input_placeholder_node.
            The content of a node can be a placeholder for concept, a constant concept, a function, or a Graph().
        (2) self.operators: contains all the names of the operators, and possibly name of the ground-node.
        (3) self.goals: set of operators that ground the input nodes based on criteria.
        (4) self.funs: A list of names for PyTorch learnable models.
        (5) self.ground_node_name: name of the ground-node. Default as None.
        (6) A set of variables for the ungrounded input nodes.

        For each node (or Graph), its input and output are OrderedDict() of Concept() instances.
        """
        self.operators = []
        self.goals = []
        self.funs = []
        self.ground_node_name = None
        super(Graph, self).__init__(G=G, **kwargs)
        if "name" in kwargs:
            self.add_operator_def(definition=kwargs)

        if G is not None:
            self.operators = deepcopy(G.operators)
            self.goals = deepcopy(G.goals)
            self.funs = G.funs
            self.ground_node_name = G.ground_node_name
            # Set up PyTorch variables and parameters for subgraph:
            for variable_name, tensor in G.get_variables().items():
                setattr(self, variable_name, nn.Parameter(tensor))
            for fun_name in G.funs:
                setattr(self, "fun_{}".format(fun_name), getattr(G, "fun_{}".format(fun_name)))

    
    def add_operator_def(self, definition):
        """Initialize the operator node from the operator definition dictionary."""
        name = definition["name"]
        if "value" in definition or "attr" in definition or "forward" in definition:
            self.operators.append(name)
            # Add body node:
            if "value" in definition:
                # The node itself is a value node (in case of input):
                self.add_node(name, value=definition["value"], type="self")
            else:
                # The node is a operator node:
                self.add_node(name, type="self")

            if "repr" in definition:
                self.nodes[name]["repr"] = nn.Parameter(definition["repr"])

            if "forward" in definition:
                fun = definition["forward"]["fun"]
                if isinstance(fun, nn.Module) and not isinstance(fun, BaseGraph):
                    self.funs.append(name)
                    setattr(self, "fun_{}".format(name), fun)
                    self.add_node(name, value=getattr(self, "fun_{}".format(name)))
                else:
                    self.add_node(name, value=fun)

                # Check if it is a criteria node:
                if name.startswith("Cri"):
                    self.goals.append(name)
                for i, arg in enumerate(definition["forward"]["args"]):
                    self.add_node("{}-{}:{}".format(name, i + 1, str(arg.mode).split("-")[0]), value=arg, type="fun-in")
                    self.add_edge("{}-{}:{}".format(name, i + 1, str(arg.mode).split("-")[0]), name, type="intra")
                    self.add_edge(name, "{}-{}:{}".format(name, i + 1, str(arg.mode).split("-")[0]), type="b-intra")
                self.add_node("{}-o:{}".format(name, str(definition["forward"]["output"].mode).split("-")[0]), 
                              value=definition["forward"]["output"], type="fun-out")
                self.add_edge("{}-o:{}".format(name, str(definition["forward"]["output"].mode).split("-")[0]), name, type="b-intra")
                self.add_edge(name, "{}-o:{}".format(name, str(definition["forward"]["output"].mode).split("-")[0]), type="intra")

                # Create and connect input placeholder nodes:
                jj = 1
                input_type_dict = {}
                for placeholder in definition["forward"]["args"]:
                    if Placeholder(DEFAULT_OBJ_TYPE).accepts(placeholder) or                     Placeholder(DEFAULT_BODY_TYPE).accepts(placeholder):
                        input_type_dict[jj] = placeholder.mode
                        jj += 1
                self.connect_input_placeholder_nodes(input_type_dict)


    def __str__(self):
        repr_str = self.graph["name"] if "name" in self.graph else "Graph"
        # Composing content string:
        content_str = ""
        operator_list = []
        for operator_name in self.operators:
            if operator_name not in self.goals:
                operator_list.append(operator_name)
        if len(operator_list) > 0:
            content_str += "operators={}, ".format(operator_list)
        if len(self.goals) > 0:
            content_str += "goals={}, ".format(self.goals)
        if len(self.input_placeholder_nodes) > 0:
            input_str = ""
            for node in self.input_placeholder_nodes:
                tip_node = self.get_node_neighboring_inter(node)
                if tip_node is None:
                    tip_node = node
                input_str += "{}, ".format(tip_node)
            content_str += "inputs=[{}], ".format(input_str[:-2])
        if len(self.constant_concept_nodes) > 0:
            content_str += "concepts={}, ".format(self.constant_concept_nodes)
        return '{}({})'.format(repr_str, content_str[:-2])


    def __repr__(self):
        if IS_VIEW and len(self.nodes) > 0:
            self.draw()
        return self.__str__()
    
    
    def __hash__(self):
        return persist_hash(self.get_string_repr())
    
    
    def __eq__(self, other):
        return self.get_string_repr() == other.get_string_repr()
    
    
    def get_string_repr(self):
        """Get 1D string representation of the operator graph"""
        combined_dict = OrderedDict()
        if self.name in self.nodes:
            combined_dict[self.name + '_value'] = tensor_to_string(self.get_node_value(self.name)) 
        else:
            combined_dict[self.name + '_value'] = "None"
        for node_name in self.topological_sort:
            node_content = self.get_node_content(node_name)
            if isinstance(self.get_node_content(node_name), Placeholder):
                combined_dict[node_name + '_inplace'] = node_content.get_inplace()
                combined_dict[node_name + '_selector'] = str(node_content.get_selector())
                combined_dict[node_name + '_value'] = tensor_to_string(self.get_node_value(node_name)) 
            elif isinstance(self.get_node_content(node_name), Concept) or isinstance(self.get_node_content(node_name), Graph):
                # Both Concept and Graph have get_string_repr implemented
                combined_dict[node_name] = node_content.get_string_repr()
        string = ""
        for key, item in combined_dict.items():
            string += "%{}!{}".format(key, item)
        return string


    def set_ground_node(self, node_name):
        """Setting up the name for ground node to expect."""
        self.ground_node_name = node_name
        return self


    def set_selector(self, selector, node_name=None):
        """Set selector on a fun-in node."""
        node_name = self.get_node_name(node_name)
        node_type = self.get_node_type(node_name)
        if node_type == "self":
            node_name = self.parent_nodes(node_name)[0]  # Get the first in_node of the operator
        assert self.get_node_type(node_name) == "fun-in", "Selector must be put in 'fun-in' node!"
        placeholder = self.get_node_content(node_name)
        assert isinstance(placeholder, Placeholder)
        placeholder.set_selector(selector)
        return self


    def get_selector(self, node_name=None):
        """Get selector on a fun-in node."""
        node_name = self.get_node_name(node_name)
        node_type = self.get_node_type(node_name)
        if node_type == "self":
            node_names = self.parent_nodes(node_name)  # Get the first in_node of the operator
        else:
            node_names = [node_name]
        selector_dict = {}
        for node_name in node_names:
            assert self.get_node_type(node_name) == "fun-in"
            placeholder = self.get_node_content(node_name)
            assert isinstance(placeholder, Placeholder)
            selector = placeholder.get_selector()
            if selector is not None:
                selector_dict[node_name] = selector
        return selector_dict


    def get_selectors(self):
        """Return a dictionary of {op_name: selectors}."""
        selectors_dict = {}
        for op_name in self.operators:
            selector_dict = self.get_selector(op_name)
            selectors_dict.update(selector_dict)
        return selectors_dict


    def set_ebm_dict(self, ebm_dict):
        """Set all the selector's ebm_dict attribute to the given ebm_dict."""
        from reasoning.experiments.models import to_ebm_models
        ebm_dict = to_ebm_models(ebm_dict)
        selectors = self.get_selectors()
        for key in selectors:
            selectors[key].ebm_dict = ebm_dict
        return self


    def remove_ebm_dict(self):
        """Remove attribute of ebm_dict from all the selectors."""
        selectors = self.get_selectors()
        for key in selectors:
            delattr(selectors[key], "ebm_dict")
        return self


    def set_cache_forward(self, cache_forward):
        """Set the cache_forward attribute for each selector in the graph."""
        selectors = self.get_selectors()
        for key in selectors:
            selectors[key].set_cache_forward(cache_forward)
        return self


    def add_op_to_selector(self, op_to_add, op_in, *opsc, **kwargs):
        """
        Args:
            op_to_add: can either be a concept or a relation operator
            op_in: fun-in nodes or self nodes that has only one fun-in
            *opsc: additional nodes in the selector to connect to/merge with.
        """
        op_in = self.get_node_name(op_in)
        op_in_type = self.get_node_type(op_in)
        if op_in_type == "self":
            node_names = self.parent_nodes(op_in)  # Get the first in_node of the operator
            assert len(node_names) == 1, "The op_in given must be a fun-in node or a self node that has only one fun-in node."
            op_in = node_names[0]
        assert self.get_node_type(op_in) == "fun-in"
        placeholder = self.get_node_content(op_in)
        assert isinstance(placeholder, Placeholder)
        selector = placeholder.get_selector()
        if selector is None:
            # The selector is not yet existing:
            num_colors = 10
            if isinstance(op_to_add, Concept):
                # op_to_add is a concept:
                selector = Concept_Pattern(
                    name=None,
                    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
                    attr={"obj_0": Placeholder(op_to_add.name)},
                    is_all_obj=True,
                    is_ebm=True,
                    is_default_ebm=False,
                    ebm_dict=kwargs["ebm_dict"],
                    CONCEPTS=kwargs["CONCEPTS"],
                    OPERATORS=kwargs["OPERATORS"],
                    device=self.device,
                    cache_forward=kwargs["cache_forward"],
                    in_channels=kwargs["in_channels"],
                    # EBM specific:
                    z_mode=kwargs["z_mode"],
                    z_first=kwargs["z_first"],
                    z_dim=kwargs["z_dim"],
                    w_type=kwargs["w_type"],
                    mask_mode=kwargs["mask_mode"],
                    aggr_mode=kwargs["aggr_mode"],
                    pos_embed_mode=kwargs["pos_embed_mode"],
                    is_ebm_share_param=kwargs["is_ebm_share_param"],
                    is_relation_z=kwargs["is_relation_z"],
                    img_dims=kwargs["img_dims"],
                    is_spec_norm=kwargs["is_spec_norm"],
                    # Selector specific:
                    channel_coef=kwargs["selector_channel_coef"],
                    empty_coef=kwargs["selector_empty_coef"],
                    obj_coef=kwargs["selector_obj_coef"],
                    mutual_exclusive_coef=kwargs["selector_mutual_exclusive_coef"],
                    pixel_entropy_coef=kwargs["selector_pixel_entropy_coef"],
                    pixel_gm_coef=kwargs["selector_pixel_gm_coef"],
                    iou_batch_consistency_coef=kwargs["selector_iou_batch_consistency_coef"],
                    iou_concept_repel_coef=kwargs["selector_iou_concept_repel_coef"],
                    iou_relation_repel_coef=kwargs["selector_iou_relation_repel_coef"],
                    iou_relation_overlap_coef=kwargs["selector_iou_relation_overlap_coef"],
                    iou_attract_coef=kwargs["selector_iou_attract_coef"],
                    connected_coef=kwargs["selector_connected_coef"],
                    SGLD_is_anneal=kwargs["selector_SGLD_is_anneal"],
                    SGLD_is_penalize_lower=kwargs["selector_SGLD_is_penalize_lower"],
                    SGLD_mutual_exclusive_coef=kwargs["selector_SGLD_mutual_exclusive_coef"],
                    SGLD_pixel_entropy_coef=kwargs["selector_SGLD_pixel_entropy_coef"],
                    SGLD_pixel_gm_coef=kwargs["selector_SGLD_pixel_gm_coef"],
                    SGLD_iou_batch_consistency_coef=kwargs["selector_SGLD_iou_batch_consistency_coef"],
                    SGLD_iou_concept_repel_coef=kwargs["selector_SGLD_iou_concept_repel_coef"],
                    SGLD_iou_relation_repel_coef=kwargs["selector_SGLD_iou_relation_repel_coef"],
                    SGLD_iou_relation_overlap_coef=kwargs["selector_SGLD_iou_relation_overlap_coef"],
                    SGLD_iou_attract_coef=kwargs["selector_SGLD_iou_attract_coef"],
                    lambd_start=kwargs["selector_lambd_start"],
                    lambd=kwargs["selector_lambd"],
                    image_value_range=kwargs["selector_image_value_range"],
                    w_init_type=kwargs["selector_w_init_type"],
                    indiv_sample=kwargs["selector_indiv_sample"],
                    step_size=kwargs["selector_step_size"],
                    step_size_img=kwargs["selector_step_size_img"],
                    step_size_z=kwargs["selector_step_size_z"],
                    step_size_zgnn=kwargs["selector_step_size_zgnn"],
                    step_size_wtarget=kwargs["selector_step_size_wtarget"],
                    connected_num_samples=kwargs["selector_connected_num_samples"],
                )
                placeholder.set_selector(selector)

            elif isinstance(op_to_add, Graph):
                # op_to_add is a relation operator:
                selector = Concept_Pattern(
                    name=None,
                    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
                    attr={
                        "obj_0": Placeholder(DEFAULT_OBJ_TYPE),
                        "obj_1": Placeholder(DEFAULT_OBJ_TYPE),
                    },
                    re={("obj_0", "obj_1"): op_to_add.name},
                    is_all_obj=True,
                    is_ebm=True,
                    is_default_ebm=False,
                    ebm_dict=kwargs["ebm_dict"],
                    CONCEPTS=kwargs["CONCEPTS"],
                    OPERATORS=kwargs["OPERATORS"],
                    device=self.device,
                    cache_forward=kwargs["cache_forward"],
                    in_channels=kwargs["in_channels"],
                    # EBM specific:
                    z_mode=kwargs["z_mode"],
                    z_first=kwargs["z_first"],
                    z_dim=kwargs["z_dim"],
                    w_type=kwargs["w_type"],
                    mask_mode=kwargs["mask_mode"],
                    aggr_mode=kwargs["aggr_mode"],
                    pos_embed_mode=kwargs["pos_embed_mode"],
                    is_ebm_share_param=kwargs["is_ebm_share_param"],
                    is_relation_z=kwargs["is_relation_z"],
                    img_dims=kwargs["img_dims"],
                    is_spec_norm=kwargs["is_spec_norm"],
                    # Selector specific:
                    channel_coef=kwargs["selector_channel_coef"],
                    empty_coef=kwargs["selector_empty_coef"],
                    obj_coef=kwargs["selector_obj_coef"],
                    mutual_exclusive_coef=kwargs["selector_mutual_exclusive_coef"],
                    pixel_entropy_coef=kwargs["selector_pixel_entropy_coef"],
                    pixel_gm_coef=kwargs["selector_pixel_gm_coef"],
                    iou_batch_consistency_coef=kwargs["selector_iou_batch_consistency_coef"],
                    iou_concept_repel_coef=kwargs["selector_iou_concept_repel_coef"],
                    iou_relation_repel_coef=kwargs["selector_iou_relation_repel_coef"],
                    iou_relation_overlap_coef=kwargs["selector_iou_relation_overlap_coef"],
                    iou_attract_coef=kwargs["selector_iou_attract_coef"],
                    connected_coef=kwargs["selector_connected_coef"],
                    SGLD_is_anneal=kwargs["selector_SGLD_is_anneal"],
                    SGLD_is_penalize_lower=kwargs["selector_SGLD_is_penalize_lower"],
                    SGLD_mutual_exclusive_coef=kwargs["selector_SGLD_mutual_exclusive_coef"],
                    SGLD_pixel_entropy_coef=kwargs["selector_SGLD_pixel_entropy_coef"],
                    SGLD_pixel_gm_coef=kwargs["selector_SGLD_pixel_gm_coef"],
                    SGLD_iou_batch_consistency_coef=kwargs["selector_SGLD_iou_batch_consistency_coef"],
                    SGLD_iou_concept_repel_coef=kwargs["selector_SGLD_iou_concept_repel_coef"],
                    SGLD_iou_relation_repel_coef=kwargs["selector_SGLD_iou_relation_repel_coef"],
                    SGLD_iou_relation_overlap_coef=kwargs["selector_SGLD_iou_relation_overlap_coef"],
                    SGLD_iou_attract_coef=kwargs["selector_SGLD_iou_attract_coef"],
                    lambd_start=kwargs["selector_lambd_start"],
                    lambd=kwargs["selector_lambd"],
                    image_value_range=kwargs["selector_image_value_range"],
                    w_init_type=kwargs["selector_w_init_type"],
                    indiv_sample=kwargs["selector_indiv_sample"],
                    step_size=kwargs["selector_step_size"],
                    step_size_img=kwargs["selector_step_size_img"],
                    step_size_z=kwargs["selector_step_size_z"],
                    step_size_zgnn=kwargs["selector_step_size_zgnn"],
                    step_size_wtarget=kwargs["selector_step_size_wtarget"],
                    connected_num_samples=kwargs["selector_connected_num_samples"],
                )
                placeholder.set_selector(selector)
            else:
                raise Exception("op_to_add {} must be a Concept or a Graph class.".format(op_to_add))
        else:
            if isinstance(op_to_add, Concept):
                selector.add_obj(op_to_add, *opsc)
            elif isinstance(op_to_add, Graph):
                selector.add_relation_manual(op_to_add.name, *opsc)
            else:
                raise Exception("op_to_add {} must be a Concept or a Graph class.".format(op_to_add))
        return self


    def set_inplace(self, inplace, node_name=None):
        """Set 'inplace' attribute on a fun-in node."""
        node_name = self.get_node_name(node_name)
        node_type = self.get_node_type(node_name)
        if node_type == "self":
            node_name = self.parent_nodes(node_name)[0]  # Get the first in_node of the operator
        assert self.get_node_type(node_name) == "fun-in", "Selector must be put in 'fun-in' node!"
        placeholder = self.get_node_content(node_name)
        assert isinstance(placeholder, Placeholder)
        placeholder.set_inplace(inplace)
        return self


    def get_inplace(self, node_name=None):
        """Get 'inplace' attribute on a fun-in node."""
        node_name = self.get_node_name(node_name)
        node_type = self.get_node_type(node_name)
        if node_type == "self":
            node_name = self.parent_nodes(node_name)[0]  # Get the first in_node of the operator
        assert self.get_node_type(node_name) == "fun-in", "Selector must be put in 'fun-in' node!"
        placeholder = self.get_node_content(node_name)
        assert isinstance(placeholder, Placeholder)
        return placeholder.get_inplace()


    ########################################
    # Forward, loss and infer functions:
    ########################################
    def forward(self, *inputs, **kwargs):
        """Perform the forward function on the graph, considering all goal nodes."""
        # No goal_optm nodes:
        if len(self.goals_optm) == 0:
            return self.forward_simple(*inputs, **kwargs)

        # Get the input keys:
        inputs, input_keys = broadcast_inputs(inputs)

        # Build graph combining main graph and goal nodes:
        G, (goal_ctrl_nodes, goal_output_nodes) = self.incorporate_goal_subgraph()

        # Prune the parts that are after the latest goal nodes:
        G.preserve_subgraph(goal_output_nodes)

        # Define loss_fun:
        shapes = [CONCEPTS[c.split(":")[1]].get_node_content(c.split(":")[1]).mode.shape for c in goal_ctrl_nodes]
        score_mode = kwargs["score_mode"] if "score_mode" in kwargs else "prod"
        kwargs2 = deepcopy(kwargs)
        kwargs2["isplot"] = False

        def loss_fun(search_variable_values):
            G.set_variable_values(goal_ctrl_nodes, search_variable_values, shapes, input_keys)
            results = G.forward_simple(*inputs, is_output_all=True, is_output_tensor=True, **kwargs2)
            score = 0
            for goal_node in goal_output_nodes:
                result = results[goal_node]
                score_goal = torch.stack(list(result.values())).float()
                if score_mode == "prod":
                    score_goal = score_goal.prod()
                elif score_mode == "sum":
                    score_goal = score_goal.mean()
                else:
                    raise
                score = score + score_goal
            return -score

        optms = self.search(goal_ctrl_nodes, loss_fun,
                            method=kwargs["method"] if "method" in kwargs else "differential_evolution",
                            example_keys=input_keys,
                           )

        # Use the best value:
        argmin = optms[0]
        self.set_variable_values(goal_ctrl_nodes, argmin, shapes, example_keys=input_keys)
        return self.forward_simple(*inputs, **kwargs)


    def forward_simple(
        self,
        *inputs,
        is_output_all=False,
        is_input_all=False,
        is_selector_all=False,
        is_output_tensor=False,
        is_NN_pathway=False,
        **kwargs
    ):
        """Perform the forward function on the main graph (without considering goal nodes).

        It first connect the self (program) DAG with the input graph, then according to the topological sort,
        performs operation along the DAG, and save the results in all operator outputs to "result" dict.
        
        Args:
            inputs: a list of OrderedDict(), where each item of the OrderedDict() is a Concept() graph. 
                Alternatively, each element of the inputs can be a Concept() graph, which will be assigned {0: Concept()}.
            is_output_all: if True, will return all the intermediate concept graph at the outnodes.
            is_input_all: if True, will record all the input values for all intermediate nodes.
        Returns:
            A list of OrderedDict()
        """
        def find_ready_goal_node(results, goal_node_dict, goal_node_inspected):
            """Check if the current results contains necessary output for the goal"""
            goal_node = None
            for node, ancestors in goal_node_dict.items():
                if node not in goal_node_inspected:
                    if set(ancestors).issubset(list(results.keys())):
                        goal_node = node
                        break
            return goal_node

        def filter_preds_by_Bool(preds, goal_preds):
            """Filter preds by Boolean."""
            preds_valid = OrderedDict()
            for goal_key, goal_value in goal_preds.items():
                goal_key_core = (goal_key,) if not isinstance(goal_key, tuple) else goal_key
                if isinstance(goal_value, Concept):
                    goal_value = goal_value.get_root_value()
                if goal_value:
                    # Find the pred_key corresponding to the goal_key:
                    for pred_key in preds:
                        pred_key_core = (pred_key,) if not isinstance(pred_key, tuple) else pred_key
                        if pred_key_core == goal_key_core[:len(pred_key_core)]:
                            break

                    preds_valid[pred_key] = preds[pred_key]
            return preds_valid

        # Connect the input graph with self (program) DAG:
        self.connect_input_placeholder_nodes()
        
        # If graph is empty, return input:
        if len(self.operators_full) == len(self.input_placeholder_nodes):
            if is_output_all:
                if is_selector_all:
                    return OrderedDict([[self.get_node_name("input-{}".format(kk + 1)), input] for kk, input in enumerate(inputs)]), {"results_selector": OrderedDict([[self.get_node_name("input-{}".format(kk + 1)), OrderedDict()] for kk, input in enumerate(inputs)])}
                else:
                    return OrderedDict([[self.get_node_name("input-{}".format(kk + 1)), input] for kk, input in enumerate(inputs)])
            else:
                if is_selector_all:
                    return inputs[0] if len(inputs) == 1 else inputs, {"results_selector": {"results_selector": OrderedDict([[self.get_node_name("input-{}".format(kk + 1)), OrderedDict()] for kk, input in enumerate(inputs)])}}
                else:
                    return inputs[0] if len(inputs) == 1 else inputs

        # Check if input is in not in the format of dictionary:
        is_dict = False
        for input_arg in inputs:
            if isinstance(input_arg, dict):
                is_dict = True
                break
        if not is_dict:
            inputs = [{0: input_arg} for input_arg in inputs]

        # Broadcast input keys, and record inputs into results:
        results = OrderedDict()
        if is_input_all:
            results_input = OrderedDict()
        if is_selector_all:
            results_selector = OrderedDict()
        input_key_list_all = []
        for i, input_arg in enumerate(inputs):
            input_node_name = self.get_node_name("input-{}".format(i + 1))
            results[input_node_name] = input_arg
            if is_selector_all:
                results_selector[input_node_name] = OrderedDict()
            if isinstance(input_arg, dict):
                input_key_list_all.append(input_arg.keys())
            else:
                input_key_list_all.append(None)
        input_key_dict = broadcast_keys(input_key_list_all)
        input_keys = list(input_key_dict.keys())

        # Initialize the value in constant concept nodes into the results:
        for node_name in self.constant_concept_nodes:
            results[node_name] = OrderedDict([[key, self.get_node_content(node_name)] for key in input_keys])

        # Initialize variables:
        self.init_variable(input_keys)

        # Obtain all the operators along the topological sort, excluding goal operators and ground_nodes:
        out_nodes_dag = self.get_output_nodes(types=["fun-out", "attr"], dangling_mode="possible", allow_goal_node=False)

        # Obtain the goal nodes without optm and their ancestors:
        goal_node_dict = self.goal_node_ancestors(is_optm=False)
        goal_node_inspected = []

        # Perform operation along the DAG:
        for out_node in out_nodes_dag:
            # Obtain the function held in the operator:
            parent_node = self.parent_nodes(out_node)
            assert len(parent_node) == 1
            parent_node = parent_node[0]
            if self[parent_node][out_node][0]["type"] == "intra-attr":
                # Obtain the attribute of parent node:
                assert parent_node in results, "The node '{}' must be in the results to get attributes!".format(parent_node)
                results[out_node] = OrderedDict()
                for key, concept in results[parent_node].items():
                    if not isinstance(concept, Concept):
                        concept = CONCEPTS[parent_node.split(":")[-1]].copy().set_node_value(concept)
                        concept.compute_attr_value()
                    out_node_proper_name = get_attr_proper_name(concept, out_node)
                    result = concept.get_attr(out_node_proper_name)
                    if result is not None:
                        results[out_node][key] = result
                    else:
                        fun = concept.nodes(data=True)[out_node_proper_name]["fun"]
                        if fun is not None:
                            results[out_node][key] = fun(concept.get_root_value())
                        else:
                            print("No function to calculate the {} of {} for {}!".format(out_node.split("^")[1], parent_node, key))
                            results[out_node][key] = None
                    if is_selector_all:
                        results_selector[out_node] = OrderedDict()

            else:
                # Use operator to obtain output concept from input concepts:
                assert self.get_node_type(parent_node) == "self", "If not getting attribute, the parent node must be a operator!"
                function = self.get_node_content(parent_node)
                has_selector = False
                
                results[out_node] = OrderedDict()
                in_nodes = self.parent_nodes(parent_node)  # in_nodes for the operator
                in_value_nodes = []  # out_nodes that feed into the in_nodes
                key_list_all = []
                in_node_mapping = {}
                kk = 0
                for jj, in_node in enumerate(in_nodes):
                    in_value_node = self.parent_nodes(in_node)
                    if len(in_value_node) > 0:
                        if "multi*" not in in_node:
                            assert len(in_value_node) == 1
                            in_value_nodes.append(in_value_node[0])
                            key_list_all.append(results[in_value_node[0]].keys() if isinstance(results[in_value_node[0]], dict) else None)
                            in_node_mapping[kk] = jj
                            kk += 1
                        else:
                            in_value_nodes += in_value_node
                            for kk_plus, in_value_node_ele in enumerate(in_value_node):
                                key_list_all.append(results[in_value_node_ele].keys() if isinstance(results[in_value_node[0]], dict) else None)
                                in_node_mapping[kk + kk_plus] = jj
                            kk += len(in_value_node)
                    else:
                        in_value_nodes.append((in_node,))  # There is no in_value_node. Use the ungrounded variable in in_node.
                        key_list_all.append(input_keys)
                        in_node_mapping[kk] = jj
                        kk += 1
                key_dict = broadcast_keys(key_list_all)
                if is_input_all:
                    results_input[out_node] = []
                if is_selector_all:
                    results_selector[out_node] = OrderedDict()
                for expanded_key, key_list in key_dict.items():
                    # Obtain input_values:
                    input_values = []

                    k = 0  # k counts the number of intermediate results (not ungrounded variables)
                    for i, (key, in_value_node) in enumerate(zip(key_list, in_value_nodes)):
                        if not isinstance(in_value_node, tuple):
                            # Get value from results calculated from previous steps:
                            if key is None:
                                input_value = results[in_value_node]
                                if is_input_all:
                                    if len(results_input[out_node]) == k:
                                        results_input[out_node].append(OrderedDict())
                                    results_input[out_node][k] = input_value
                            else:
                                input_value = results[in_value_node][key]
                                # Record input_value:
                                if is_input_all:
                                    if len(results_input[out_node]) == k:
                                        results_input[out_node].append(OrderedDict())
                                    results_input[out_node][k][key] = input_value
                            k += 1
                        else:
                            # Get value from ungrounded concepts:
                            input_value = getattr(self, "variable_{}_{}".format(in_value_node[0], key))

                        # If the first in_node has a selector, use the selector to select the input_value:
                        selector_innode = self.get_selector(in_nodes[in_node_mapping[i]])
                        if len(selector_innode) > 0:
                            selector = selector_innode[next(iter(selector_innode))]
                            has_selector = True
                            if is_NN_pathway:
                                input_value, _ = selector.forward_NN(input_value)
                            else:
                                input_value_ori = deepcopy(input_value)
                                inp_name_ori = input_value_ori.get_node_name()
                                # If selector applies to a System, must apply the operator to the system concept
                                if "System" in inp_name_ori:
                                    input_value = input_value.get_refer_subconcept(selector)
                                else:
                                    input_value = input_value.get_refer_nodes(selector)

                        # Append:
                        input_values.append(input_value)

                    # Execute the operator:
                    if not has_selector:
                        output_value = function(*input_values, is_NN_pathway=is_NN_pathway)
                    else:
                        # Perform operator on selected objects (either inplace or use new instance.)
                        if is_NN_pathway:
                            output_value = function(*input_values, is_NN_pathway=is_NN_pathway)
                        else:
                            obj_trans_dict = {}
                            inplace = self.get_inplace(in_nodes[0])
                            inp_name_ori = input_value_ori.get_node_name()
                            if "System" in inp_name_ori:
                                output_value = function(input_values[0])
                            elif len(input_values[0]) > 0:
                                for obj_name, obj in input_values[0].items():
                                    obj_trans = function(obj, *input_values[1:], is_NN_pathway=is_NN_pathway)
                                    obj_trans_dict[obj_name] = obj_trans
                                    if inplace:
                                        # Perform operation on the object, but keep the obj_name unchanged:
                                        input_value_ori.remove_attr(obj_name, change_root=True)
                                        input_value_ori.add_obj(obj_trans, obj_name=obj_name, change_root=True)
                                if inplace:
                                    output_value = input_value_ori
                                else:
                                    if obj_name.split(":")[-1] == DEFAULT_OBJ_TYPE:
                                        output_value, _ = get_comp_obj(obj_trans_dict, CONCEPTS)
                                    else:
                                        output_value = obj_trans_dict
                            else:
                                if inplace:
                                    output_value = input_value_ori
                                else:
                                    output_value = CONCEPTS[DEFAULT_OBJ_TYPE].copy().set_node_value(torch.zeros(1, 1)).set_node_value([0, 0, 1, 1], "pos")
                        # Record selector info:
                        if is_selector_all:
                            results_selector[out_node][expanded_key] = {"operated": list(input_values[0].keys()),
                                                                        "from": in_value_nodes,
                                                                        "inplace": inplace}

                    # Formatting output:
                    if isinstance(output_value, bool):
                        output_value = torch.BoolTensor([output_value])[0].to(self.device)
                    if is_output_tensor and isinstance(output_value, Concept):
                        output_value = output_value.get_root_value()
                    # Record output_value:
                    if isinstance(output_value, dict):
                        for j, output_value_ele in output_value.items():
                            new_key = expanded_key + (j,) if isinstance(expanded_key, tuple) else (expanded_key, j)
                            results[out_node][new_key] = output_value_ele
                    else:
                        results[out_node][expanded_key] = output_value
                # Make sure that all keys have the same length for all examples:
                results[out_node] = canonicalize_keys(results[out_node])

                # Check goal_nodes without optm, and reduce number of predictions if determined by goal nodes:
                goal_node = find_ready_goal_node(results, goal_node_dict, goal_node_inspected)
                if goal_node is not None:
                    goal_input_nodes = self.parent_nodes(goal_node)
                    goal_in_value_nodes = [self.parent_nodes(node)[0] for node in goal_input_nodes]
                    goal_input_values = [results[node] for node in goal_in_value_nodes]
                    goal_preds = self.get_node_content(goal_node)(*goal_input_values)
                    results[out_node] = filter_preds_by_Bool(results[out_node], goal_preds)

        # Return results:
        info_all = {}
        if is_input_all:
            info_all["results_input"] = results_input
        if is_selector_all:
            info_all["results_selector"] = results_selector
        if is_output_all:
            if is_input_all or is_selector_all:
                return results, info_all
            else:
                return results
        else:
            output_mode = out_nodes_dag[-1].split(":")[-1]
            output_dict = OrderedDict()
            for key, item in results[out_nodes_dag[-1]].items():
                if is_output_tensor:
                    if isinstance(item, Concept):
                        item = item.get_root_value()
                else:
                    if not isinstance(item, Concept):
                        item = CONCEPTS[output_mode].copy().set_node_value(item, output_mode)
                if "isplot" in kwargs and kwargs["isplot"]:
                    item_value = item.get_root_value() if isinstance(item, Concept) else item
                    if len(item_value.shape) == 2:
                        visualize_matrices([item_value])
                output_dict[key] = item

            if not is_dict and len(output_dict) == 1:
                output_dict = output_dict[0]
            if is_input_all or is_selector_all:
                return output_dict, info_all
            else:
                return output_dict


    def get_score(self, inputs, targets, score_fun, preds=None, variable_dict=None, return_preds=False, **kwargs):
        """Obtain the score and the corresponding best idx for matching pred and target.
        score must be between 0 and 1.
        
        Args:
            inputs: a list of Concept() or OrderedDict() of Concept().
            targets: a Concept() or OrderedDict() of Concept()
            score_fun: score function to evaluate the score for each pair of pred and target.
            variable_dict: a dictionary of {key: variable_tensor}, or a OrderedDict() of such dictionaries.
                If None, will use current self's variables.
        Returns:
            A score (if pred is a Concept()) or OrderedDict() of scores.
        """
        if preds is None:
            if not isinstance(inputs, list):
                inputs = [inputs]
            preds = self(*inputs)
        else:
            assert inputs is None
        if variable_dict is not None and not isinstance(variable_dict[list(variable_dict.keys())[0]], dict):
            # Set variable only for the case where variable_dict is a dictionary of {key: variable_tensor}:
            self.set_variables(variable_dict)

        if not isinstance(preds, dict):
            assert not isinstance(targets, dict)
            score, _ = score_fun(preds, targets)
            return score
        else:
            score_dict = OrderedDict()
            if not isinstance(targets, dict):
                if variable_dict is not None:
                    key_dict = broadcast_keys([list(preds.keys()), list(variable_dict.keys())])
                    preds = []
                    for new_key, (pred_key, variable_key) in key_dict.items():
                        self.set_variables(variable_dict[variable_key])
                        inputs_ele = [OrderedDict([[pred_key, inputs[i][pred_key]]]) for i in range(len(inputs))]
                        pred = self(*inputs_ele)
                        preds.append(pred)
                        score, _ = score_fun(pred, targets)
                        score_dict[key] = score
                else:
                    for key, pred in preds.items():
                        score, _ = score_fun(pred, targets)
                        score_dict[key] = score
            else:
                # Both preds and targets are dict:
                if variable_dict is not None:
                    key_dict = broadcast_keys([list(preds.keys()), list(targets.keys()), list(variable_dict.keys())])
                    preds = []
                    for new_key, (pred_key, target_key, variable_key) in key_dict.items():
                        self.set_variables(variable_dict[variable_key])
                        inputs_ele = [inputs[i][pred_key] for i in range(len(inputs))]
                        pred = self(*inputs_ele)
                        preds.append(pred)
                        score, _ = score_fun(pred, targets[target_key])
                        score_dict[new_key] = score
                else:
                    key_dict = broadcast_keys([list(preds.keys()), list(targets.keys())])
                    for new_key, (pred_key, target_key) in key_dict.items():
                        score, _ = score_fun(preds[pred_key], targets[target_key])
                        score_dict[new_key] = score
        if return_preds:
            return score_dict, preds
        else:
            return score_dict


    def get_loss(self, inputs, targets, score_fun, **kwargs):
        """Obtain the loss, with the same input and output APIs as get_score, except that loss = 1 - score."""
        score_dict = self.get_score(inputs, targets, score_fun=score_fun, **kwargs)
        if not isinstance(score_dict, dict):
            return 1 - score_dict
        loss_dict = OrderedDict()
        for key, score in score_dict.items():
            loss_dict[key] = 1 - score
        return loss_dict


    def get_loss_NN(self, inputs, targets, loss_fun, reduction="mean"):
        """Compute the loss in the neural network (NN) pathway, using the given loss_fun."""
        # Make sure that the key matches:
        assert isinstance(inputs, list)
        for input in inputs:
            assert input.keys() == targets.keys()
        # Compute loss for each example:
        loss_dict = OrderedDict()
        for key, target in targets.items():
            pred = self.forward(*[inputs[i][key] for i in range(len(inputs))], is_NN_pathway=True, is_output_tensor=True)
            loss_dict[key] = loss_fun(pred, target)
        if reduction == "none":
            return loss_dict
        elif reduction == "mean":
            return torch.stack(list(loss_dict.values())).mean()
        elif reduction == "sum":
            return torch.stack(list(loss_dict.values())).sum()
        else:
            raise


    def search(self, search_variable_list, loss_fun, example_keys=None, method=None):
        """Find the value of each variable in search_variable_list that minimizes its
        corresponding loss function.

        Parameters:
            search_variable_list: the name of variables whose values should be argmin of 
                loss_function; e.g. search_variable_list = ["variable_S-2:Pos", "variable_S-2:Color"]

            method: brute-force search: "brute"; 
                    random search: "basinhopping";
                    heuristic search: "differential_evolution"

        Returns:
            optms: a 2-element list.
            optms[0]: A 1-D array containing the coordinates of points at which the objective 
                      function had its minimum value; 
            optms[1]: Function value at the optimal point.
        """

        method = "differential_evolution" if method is None else method
        optms = []
        ranges = ()
        x0 = []; xmax = []; xmin = []
        length = len(example_keys) if example_keys is not None else 1
        for variable in search_variable_list:
            this_operator = variable.split("_")[0].split(":")[1] # obtain the operator in each variable, e.g. "Pos"
            this_operator_mode = CONCEPTS[this_operator].get_node_content(this_operator).mode
            this_shape = int(np.prod(this_operator_mode.shape) * length) # e.g. (4,6) with 2 example keys will become 4 * 6 * 2
            this_range = slice(this_operator_mode.range[0], this_operator_mode.range[-1] + 1, 1)
            ranges = ranges + tuple([this_range] * this_shape)
            x0 = x0 + [1.] * this_shape
            xmax = xmax + [this_operator_mode.range[-1] + 1] * this_shape
            xmin = xmin + [this_operator_mode.range[0]] * this_shape

        if method == "brute":
            optms = optimize.brute(loss_fun,
                                   ranges,
                                   full_output=True,
                                   finish=optimize.fmin)

        elif method == "basinhopping":
            minimizer_kwargs = {"method": "BFGS"}
            mybounds = MyBounds(xmax, xmin)
            vals = optimize.basinhopping(loss_fun, 
                                         x0, 
                                         stepsize=1.0, 
                                         minimizer_kwargs=minimizer_kwargs, 
                                         niter=2000, accept_test=mybounds)
            optms = [vals.x, vals.fun]

        elif method == "differential_evolution": # best method so far
            bounds = list(zip(xmin, xmax))
            vals = optimize.differential_evolution(loss_fun, 
                                                   bounds, 
                                                   maxiter=2000)
            optms = [vals.x, vals.fun]

        else:
            raise Exception("method {} is not supported!".format(method))

        return optms


    def infer(self, inputs, targets, score_fun=None, learnable=["variable", "fun"], **kwargs):
        """Infer the best variable values for the dangling nodes, given inputs and targets.

        Args:
            inputs: a list of Concept() or OrderedDict() of Concept().
            targets: a Concept() or OrderedDict() of Concept(). The requirements for 
                the inputs and targets is that their keys are broadcastable.
            score_fun: score function to evaluate the score for each pair of pred and target.
            
        
        Returns:
            optm_dict: OrderedDict() of (argmin, score).
        """
        key_list_all = [input_arg.keys() for input_arg in inputs]
        key_list_all.append(targets.keys())
        key_dict = broadcast_keys(key_list_all)
        self.init_variable(key_dict.keys())

        optm_dict = OrderedDict()
        for expanded_key, key_list in key_dict.items():
            search_variable_list = [key for key in self.get_variables(example_key=expanded_key)]
            input_item = [input[key_list[i]] for i, input in enumerate(inputs)]
            target_item = targets[key_list[-1]]
            if len(search_variable_list) == 0:
                score = self.get_score(input_item, target_item, score_fun=score_fun)
                score_optms = [[], score]
                optm_dict[expanded_key] = score_optms

            else:
                search_variable_list = ["_".join(variable.split("_")[1:]) for variable in search_variable_list]
                search_operators = [variable.split("_")[0].split(":")[1] for variable in search_variable_list]
                shapes = [CONCEPTS[c].get_node_content(c).mode.shape for c in search_operators]
                kwargs2 = deepcopy(kwargs)
                kwargs2["isplot"] = False

                def loss_fun(search_variable_values):
                    self.set_variable_values(search_variable_list, search_variable_values, shapes)
                    loss_dict = self.get_loss(input_item, target_item, score_fun=score_fun, **kwargs2)
                    if isinstance(loss_dict, dict):
                        loss = torch.stack(list(loss_dict.values())).min()
                    else:
                        loss = loss_dict
                    return loss

                optms = self.search(search_variable_list, loss_fun,
                                    method=kwargs["method"] if "method" in kwargs else "differential_evolution",
                                   )
                self.set_variable_values(search_variable_list, optms[0], shapes)
                self.round_variables(expanded_key)
                optm_dict[expanded_key] = [self.get_variables(example_key=expanded_key), 1 - optms[1]]
                if "isplot" in kwargs and kwargs["isplot"]:
                    self(*input_item, isplot=True)

        return optm_dict

        # Note that we can also try heuristic search hill_climbing or genetic search
        # and random search https://towardsdatascience.com/hyperparameter-optimization-in-python-part-1-scikit-optimize-754e485d24fe


    def accepts(self, inputs, targets):
        """Whether the current operator's expected input and output types are compatible with the input and target."""
        key_dict = broadcast_keys([list(inputs[0].keys()), list(targets.keys())])
        for _, key_list in key_dict.items():
            input_key, target_key = key_list
            break
        target = targets[target_key]

        operator_input_modes = [node.split(":")[-1] for node, is_input in self.input_nodes.items() if is_input]
        input_modes = [input_arg[input_key].name for input_arg in inputs]
        is_valid_input = accepts(operator_input_modes, input_modes, [CONCEPTS, NEW_CONCEPTS], mode="exists")

        is_valid_target = False
        for operator_target_node in self.get_output_nodes(types=["fun-out"], dangling_mode=True):
            target_mode = split_string(target.name)[0]
            target_modes_inherit = get_inherit_modes(target_mode, CONCEPTS, type="to")
            operator_target_mode = operator_target_node.split(":")[-1]
            if operator_target_mode in target_modes_inherit:
                is_valid_target = True
                break
        return is_valid_input and is_valid_target


    ########################################
    # Get items from graph:
    ########################################
    def get_node_input(self, node_name, mode=None):
        """Obtain the input node of an operator whose mode is the same as given mode."""
        if self.get_node_type(node_name) == "self":
            parent_nodes = self.parent_nodes(node_name)
            if mode is None:
                return parent_nodes
            else:
                return [parent_node for parent_node in parent_nodes if canonical(parent_node.split(":")[-1]) == mode]
        else:
            return None


    def get_node_output(self, node_name, mode=None):
        """Obtain the output node of an operator whose mode is the same as given mode."""
        assert self.get_node_type(node_name) in ["self"]
        child_node = self.child_nodes(node_name)
        assert len(child_node) == 1
        if mode is None:
            return child_node
        else:
            return [node_name for node_name in child_node if node_name.split(":")[-1] == mode]


    @property
    def core_graph(self):
        return self


    @property
    def core_graph_shallow(self):
        return self


    @property
    def main_graph(self):
        """Return the subgraph excluding all goal nodes."""
        G = self.copy()
        G.remove_subgraph(self.goals)
        return G


    @property
    def main_forward_graph(self):
        """Return the subgraph excluding the goal nodes that involves optimization."""
        G = self.copy()
        G.remove_subgraph(self.goals_optm)
        return G


    @property
    def input_nodes(self):
        """Return input-nodes and whether they are grounded."""
        input_nodes_dict = OrderedDict()
        input_placeholder_nodes = self.input_placeholder_nodes
        constant_concept_nodes = self.constant_concept_nodes
        for operator in self.operators_core:
            nodes = self.parent_nodes(operator)
            for node in nodes:
                data = self.nodes(data=True)[node]
                if len(node.split("-")) > 1 and "-o" not in node and "input-" not in node and "concept-" not in node and "Ctrl" not in node: # is an input node
                    is_input = True
                    # Check if the node is the child of some other nodes:
                    for neighbor_node, info in self[node].items():
                        if info[0]["type"].startswith("b-"):
                            is_input = False
                            break
                    # Check if this node is fed into a goal node for grounding:
                    for child_node in self.child_nodes(node):
                        if "Ctrl" in child_node:
                            is_input = False
                            break
                    input_nodes_dict[node] = is_input
        return input_nodes_dict


    @property
    def dangling_nodes(self):
        """Return a list of input nodes that does not expect input
        (has no arrow fed to it, and does not expect to connect with ground nodes,
         thus when calling forward function must initialize a PyTorch variable.)"""
        dangling_nodes = []
        for node, is_input in self.input_nodes.items():
            variable_name = "variable_{}".format(node)
            # is_dangling is True if the node does not expect any input:
            is_dangling = True if is_input and self.get_node_content(node).mode != self.ground_node_name else False
            if is_dangling:
                dangling_nodes.append(node)
        return dangling_nodes


    @property
    def input_node_concept(self):
        """Get all the danging concept input_nodes."""
        concept_nodes = []
        for node, is_input in self.input_nodes.items():
            if is_input:
                mode = node.split(":")[-1]
                if mode == "Concept":
                    concept_nodes.append(node)
        return concept_nodes


    @property
    def is_fully_grounded(self):
        """Check if the current graph is fully grounded (have no dangling input nodes)."""
        return len(self.dangling_nodes) == 0


    @property
    def control_nodes(self):
        """Control nodes: input_nodes that is fed to the 'Ctrl' of a goal node."""
        control_nodes = []
        for node in self.input_nodes:
            for child_node in self.child_nodes(node):
                if "Ctrl" in child_node:
                    control_nodes.append(node)
        return control_nodes


    @property
    def goal_nodes(self):
        """Goal nodes. The item is True when the control node has control variables."""
        node_dict = OrderedDict()
        for node in self.goals:
            parent_nodes = self.parent_nodes(node)
            if "Cri-0:Ctrl" in parent_nodes:
                node_dict[node] = True  # Need to optimize
            else:
                node_dict[node] = False
        return node_dict


    @property
    def goals_optm(self):
        return [node for node, is_optm in self.goal_nodes.items() if is_optm]


    def goal_node_ancestors(self, node_name=None, is_optm=None):
        """Get the ancestor output nodes of a goal node."""
        node_dict = OrderedDict()
        output_nodes = self.get_output_nodes(types=["fun-out", "input", "attr"], dangling_mode="possible")
        for node, is_optm_ele in self.goal_nodes.items():
            if is_optm is not None and (is_optm_ele is not is_optm):
                continue
            if node_name is not None:
                if node != self.get_node_name(node_name):
                    continue
            ancestors = self.get_ancestors("{}-o".format(node))
            node_dict[node] = list(set(output_nodes).intersection(set(ancestors)))
        return node_dict


    @property
    def output_nodes(self):
        return self.get_output_nodes(types=["fun-out", "attr", "input"], dangling_mode="possible")


    def get_output_nodes(self, types=["fun-out", "attr", "input"], dangling_mode="possible", allow_goal_node=False):
        """Return output-nodes and whether they feed into other inputs."""
        assert set(types).issubset({"fun-out", "attr", "input"})
        output_nodes = OrderedDict() if isinstance(dangling_mode, dict) else []
        for node_name in self.topological_sort:
            if self.get_node_type(node_name) in types:
                if (not allow_goal_node) and node_name.split("-")[0] in self.goals:
                    continue
                if dangling_mode == "possible":
                    output_nodes.append(node_name)
                elif dangling_mode == "available":
                    output_nodes.append(node_name)
                elif dangling_mode in [True, False]:
                    if self.check_dangling(node_name) is dangling_mode:
                        output_nodes.append(node_name)
                elif isinstance(dangling_mode, dict):
                    if "possible" in dangling_mode:
                        output_nodes[node_name] = dangling_mode["possible"]
                    if "available" in dangling_mode and self.check_available(node_name):
                        output_nodes[node_name] = dangling_mode["available"]
                    if "dangling" in dangling_mode and self.check_dangling(node_name):
                        output_nodes[node_name] = dangling_mode["dangling"]
                else:
                    raise
        return output_nodes


    @property
    def input_placeholder_nodes(self):
        nodes = []
        for node_name in self.nodes:
            if self.get_node_type(node_name) == "input":
                nodes.append(node_name)
        return nodes


    @property
    def constant_concept_nodes(self):
        nodes = []
        for node_name in self.nodes:
            if self.get_node_type(node_name) == "concept":
                nodes.append(node_name)
        return nodes


    @property
    def operators_full(self):
        """Returns input_place_holder_nodes, operators, constant concept nodes, and relevant attribute nodes in a sorted list.
        Here the operators are the one that can hold concepts.
        """
        input_placeholder_nodes = [self.operator_name(op) for op in self.input_placeholder_nodes]
        operators = input_placeholder_nodes
        for operator in self.topological_sort:
            if len(operator.split("-")) == 1 or self.get_node_type(operator) == "concept":
                operators.append(operator)
        operators_dict = OrderedDict([[operator, [operator]] for operator in operators])
        for i, node in enumerate(self.topological_sort):
            if "input" in node and "^" in node:
                # input attribute node:
                op = node.split("^")[0]
                operators_dict[op].append(self.operator_name(node))
            elif "^" in node:
                # attribute node based on intermediate output
                op = node.split("-")[0]
                operators_dict[op].append(self.operator_name(node))
        full_operators = []
        for _, operator_expand in operators_dict.items():
            full_operators += operator_expand
        full_operators = OrderedDict([[operator, i] for i, operator in enumerate(full_operators)])
        return full_operators


    @property
    def operators_core(self):
        """Returns the operators in a sorted list."""
        operators = remove_duplicates([operator.split("-")[0] for operator in self.topological_sort if len(operator.split("-")) == 1])
        assert set(operators) == set(self.operators)
        return operators


    @property
    def reprs(self):
        """Return the reprentations for operators in N x REPR_DIM, where N is the number of
        operators in the graph, and each row corresponds to the global representation of the operator.
        """
        x = []
        for operator in self.operators_core:
            operator_repr = self.get_node_repr(operator)
            x.append(operator_repr)
        x = torch.stack(x, 0)
        return x


    @property
    def edge_index(self):
        """Return edge_index for operators in COO format, where the operators' index 
        is according to self.operators_full."""
        edge_index = []
        operators_full = self.operators_full
        for i, operator in enumerate(operators_full):
            for child_operator in self.child_operators(operator):
                j = operators_full[child_operator]
                edge_index.append([i, j])
        if len(edge_index) > 0:
            edge_index = to_Variable(edge_index).long().T.to(self.device)
            return edge_index
        else:
            return torch.zeros(2, 0).long().to(self.device)


    @property
    def edge_type(self):
        """
        edge attributes:
        3-hot: whether it is (0) generic inter-operator edge, (1) attribute edge, or (2) connect to goal node.  """
        edge_type = []
        operators_full = self.operators_full
        for i, operator in enumerate(operators_full):
            for child_operator in self.child_operators(operator):
                j = operators_full[child_operator]
                if operator.split("-")[0] == child_operator.split("-")[0]:
                    edge_type.append(1)
                elif "Cri" in child_operator:
                    # Connected to goal node:
                    edge_type.append(2)
                else:
                    edge_type.append(0)
        if len(edge_type) > 0:
            edge_type = torch.LongTensor(edge_type).to(self.device)
            return edge_type
        else:
            return torch.zeros(0).to(self.device)


    def get_op_attr(
        self,
        OPERATORS,
        inputs=None,
        targets=None,
        parse_pair_ele=None,
        op_attr_modes="0123",
        allowed_attr="obj",
        cache_dirname=None,
    ):
        """
        Obtain operator attributes amenable for PyG.

        Args:
            OPERATORS: dictionary of operators
            CONCEPTS: dictionary of concepts
            inputs: inputs for computing shape and intermediate result
            targets: OrderedDict of targets
            parse_pair_ele: function parsing the pair of inter_result and target into correspondence of component objects.
            op_attr_modes:
                0. operator type: |O| + 4 hot (4 is for input-node, concept-node, Do graph and For graph, respectively)
                1. operator state: 1-integer, number of dangling input nodes
                2. shape: 2-integer vector
                3. (intermediate-result, target) in terms of concept graph.
            allowed_attr: allowed attributes when computing the concept graph correspondence. Choose from "obj" and "all"
            cache_dirname: if not None, will save the cached files in the cache_dirname for future use.

        Covariant encoding of concept graphs:
            Each input, intermediate or target concept graph has a covariant encoding, on which the operator on top of that
            can refer to any subset of components. This is mainly used in two scenarios: pairing of intermediate result
            with the target, as well as building the selector for the downstream operators.

        Returns:
            op_attr: in a form amenable for PyG.
        """
        length_op = len(OPERATORS)
        OPERATOR_KEYS = list(OPERATORS.keys())
        operators_full = self.operators_full
        op_attr = {}

        # Get operator type as one-hot in with length len(OPERATORS) + 4:
        if "0" in op_attr_modes:
            op_types = torch.zeros(len(operators_full), length_op + 4).to(self.device)
            for i, op in enumerate(operators_full):
                if "input" in op:
                    idx = length_op
                elif "concept" in op:
                    idx = length_op + 1
                elif "Do" in op:
                    idx = length_op + 2
                elif "For" in op:
                    idx = length_op + 3
                else:
                    op_core = split_string(op.split("-")[0])[0]
                    idx = OPERATOR_KEYS.index(op_core)
                op_types[i, idx] = 1
            op_attr["op_types"] = op_types

        # Get the state the operator:
        if "1" in op_attr_modes:
            op_states = torch.zeros(len(operators_full), 1).to(self.device)
            for i, op in enumerate(operators_full):
                out_node = self.get_to_outnode(op)
                node_name = self.get_node_name(out_node)
                if self.get_node_type(node_name) in ["attr", "input", "concept"]:
                    op_states[i] = 0
                else:
                    assert self.get_node_type(op) == "self"
                    in_nodes = self.parent_nodes(op)
                    num_dangling_innodes = 0
                    for node in in_nodes:
                        if len(self.parent_nodes(node)) == 0:
                            num_dangling_innodes += 1
                    # To do: deal with grounding using goal node.
                    op_states[i] = num_dangling_innodes
            op_attr["op_states"] = op_states

        # Get the output shape of the operator:
        if "2" in op_attr_modes and inputs is not None:
            input_keys = list(inputs[0].keys())
            input_keys_tuple = [(key,) if not isinstance(key, tuple) else key for key in input_keys]
            results = self(*inputs, is_output_all=True)
            op_shapes = []
            for i, op in enumerate(operators_full):
                out_node = self.get_to_outnode(op)
                node_name = self.get_node_name(out_node)
                if "Cri" in node_name:
                    op_shape = OrderedDict([[key, torch.FloatTensor([1, 0]).to(self.device)] for key in input_keys])
                else:
                    op_shape = get_op_shape(results[node_name])
                op_shapes.append(op_shape)
            op_attr["op_shapes"] = op_shapes

        if "3" in op_attr_modes and inputs is not None and targets is not None and parse_pair_ele is not None:
            op_pairs = []
            for i, op in enumerate(operators_full):
                out_node = self.get_to_outnode(op)
                node_name = self.get_node_name(out_node)
                inter_results = results[node_name]
                pair_data_dict = {}
                for key, inter_result in inter_results.items():
                    if inter_result.name != DEFAULT_OBJ_TYPE:
                        continue
                    target = targets[key]
                    pair_x, edge_index, edge_attr = get_pair_PyG_data(
                        inter_result,
                        target,
                        OPERATORS,
                        parse_pair_ele,
                        allowed_attr=allowed_attr,
                        cache_dirname=cache_dirname,
                    )
                    pair_data_dict[key] = {"x": pair_x, "edge_index": edge_index, "edge_attr": edge_attr}
                op_pairs.append(pair_data_dict)
            op_attr["op_pairs"] = op_pairs
        return op_attr


    def get_PyG_data(
        self,
        OPERATORS,
        inputs=None,
        targets=None,
        parse_pair_ele=None,
        op_attr_modes="0123",
        allowed_attr="obj",
        cache_dirname=None,
    ):
        """Get the graph data in PyG format.
        Attributes: x: global reprenstation for each operator, where each row is for one operator 
                                sorted by self.operators_full. See self.get_op_attr() for details.
                    edge_index: edge_index for operators in COO format, where the operators' index 
                                is according to self.operators_full
        """
        from torch_geometric.data import Data
        op_attr = self.get_op_attr(
            OPERATORS=OPERATORS,
            inputs=inputs,
            targets=targets,
            parse_pair_ele=parse_pair_ele,
            op_attr_modes=op_attr_modes,
            allowed_attr=allowed_attr,
            cache_dirname=cache_dirname,
        )
        data = Data(x=op_attr, edge_index=self.edge_index, edge_type=self.edge_type)
        return data


    @property
    def operators_dangling(self):
        """Return operators that are dangling."""
        dangling_operators = []
        for operator in self.operators:
            is_dangling = True
            op_input_nodes = self.parent_nodes(operator)
            for node in op_input_nodes:
                if len(self.parent_nodes(node)) > 0:
                    is_dangling = False
            if is_dangling:
                dangling_operators.append(operator)
        return dangling_operators


    def child_operators(self, operator):
        """Get the child operators of the current operator."""
        assert operator in self.operators_full, "The operator {} does not belong to self.operators_full!".format(operator)
        child_nodes = self.child_nodes(operator)
        if len(child_nodes) == 1 and self.get_node_type(child_nodes[0]) == "fun-out":
            # child_node is an output node:
            child_nodes = self.child_nodes(child_nodes[0])
        child_operators = [self.operator_name(node) for node in child_nodes]
        return child_operators


    def parent_operators(self, operator):
        """Get the child operators of the current operator."""
        assert operator in self.operators_full, "The operator {} does not belong to self.operators_full!".format(operator)
        parent_nodes = self.parent_nodes(operator)
        if len(parent_nodes) == 1 and ("^" in parent_nodes[0] or "input" in parent_nodes[0]):
            parent_operators = [self.operator_name(parent_nodes[0])]
        else:
            parent_operators = []
            for node1 in parent_nodes:
                candidate_nodes = self.parent_nodes(node1)
                for node2 in candidate_nodes:
                    parent_operators.append(self.operator_name(node2))
        return parent_operators


    def get_op_embedding(self, op_name, OPERATORS, CONCEPTS):
        """Get the embedding of a node, used for working memory.

        embedding:
            op_type_embed: embedding for differentiating different roles in the full corr_graph.
            op_name_embed: embedding for the type of the concept or operator
            op_API_embed:  embedding for the output type of the op.
        """
        if isinstance(op_name, list):
            return torch.cat([self.get_op_embedding(ele, OPERATORS, CONCEPTS) for ele in op_name])
        else:
            op_mode = split_string(op_name.split(":")[-1])[0]
            op_type = get_op_type(op_name)
            device = self.device
            default_embed = torch.zeros(REPR_DIM).to(device)
            if op_type == "op":
                # An operator node:
                op_type_embed = torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
                op_name_embed = OPERATORS[op_mode].get_node_repr()
                op_outnode_mode = self.get_to_outnode(op_name).split(":")[-1]
                op_API_embed = CONCEPTS[op_outnode_mode].get_node_repr()
            elif op_type == "op-in":
                # An input or output node:
                op_type_embed = torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
                op_name_embed = default_embed
                op_API_embed = CONCEPTS[op_mode].get_node_repr()
            elif op_type == "op-attr":
                # An attribute node:
                op_type_embed = torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
                op_name_embed = default_embed
                op_API_embed = CONCEPTS[op_mode].get_node_repr()
            elif op_type == "op-sc":
                # A concept node in a selector:
                op_type_embed = torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
                op_mode = op_name.split(":")[-1]
                op_name_embed = CONCEPTS[op_mode].get_node_repr()
                op_API_embed = default_embed
            elif op_type == "op-so":
                # A relation in a selector:
                op_type_embed = torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
                op_mode = op_name.split(":")[-1]
                op = OPERATORS[op_mode]
                op_name_embed = op.get_node_repr()
                op_outnode_mode = op.get_to_outnode(op.name).split(":")[-1]
                op_API_embed = CONCEPTS[op_outnode_mode].get_node_repr()
            elif op_type == "op-op":
                # Operator's inner operator, e.g. "ForGraph->op$Copy":
                op_type_embed = torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).to(device)
                op_name_core = split_string(op_name.split("$")[-1])[0]
                op_name_embed = OPERATORS[op_name_core].get_node_repr()
                op_outnode_mode = self.get_to_outnode(op_name_core).split(":")[-1]
                op_API_embed = CONCEPTS[op_outnode_mode].get_node_repr()
            elif op_type == "input":
                op_type_embed = torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).to(device)
                op_name_embed = default_embed
                op_API_embed = CONCEPTS[op_mode].get_node_repr()
            elif op_type == "concept":
                op_type_embed = torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).to(device)
                op_name_embed = default_embed
                op_API_embed = CONCEPTS[op_mode].get_node_repr()
            elif op_type == "o":
                op_mode = op_name.split("o$")[-1]
                op_type_embed = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).to(device)
                op = OPERATORS[op_mode]
                op_name_embed = op.get_node_repr()
                op_outnode_mode = op.get_to_outnode(op.name).split(":")[-1]
                op_API_embed = CONCEPTS[op_outnode_mode].get_node_repr()
            elif op_type == "c":
                op_mode = op_name.split("c$")[-1]
                op_type_embed = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).to(device)
                op = CONCEPTS[op_mode]
                op_name_embed = op.get_node_repr()
                op_API_embed = default_embed
            elif op_type == "result":
                op_type_embed = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).to(device)
                op_name_embed = default_embed
                op_API_embed = CONCEPTS[op_mode].get_node_repr()
            elif op_type == "target":
                op_type_embed = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).to(device)
                op_name_embed = default_embed
                op_API_embed = CONCEPTS[op_mode].get_node_repr()
            elif op_type == "opparse":
                # In the format of "opparse$Draw->0->obj_1->RotateA":
                op_type_embed = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).to(device)
                op_mode_split = op_name.split("->")[-1].split("_")
                op_mode = split_string(op_mode_split[0])[0]
                op = OPERATORS[op_mode]
                op_name_embed = op.get_node_repr()
                if len(op_mode_split) > 1 and "changeColor" in op_mode_split:
                    op_name_embed = op_name_embed + RELATION_EMBEDDING["changeColor"].to(device)
                op_outnode_mode = op.get_to_outnode(op.name).split(":")[-1]
                op_API_embed = CONCEPTS[op_outnode_mode].get_node_repr()
            else:
                raise Exception("op_type '{}' is not supported!".format(op_type))
            embedding = torch.cat([op_type_embed, op_name_embed, op_API_embed])[None, :]
            return embedding


    def get_neighbors(self, op_name):
        """Get the neighbor nodes of the op_name given.

        Neighbor naming convention:
            N-op-parent:    operator's parent op
            N-op-child:     operator's child op
            N-op-in:        operator's input node
            N-opout-c:      operator's output's concept

            N-opin-opsc:    from selector root to its consistuent concept
            N-opin-opso:    from selector root to its consistuent relations
            N-opin-c:       operator's input's concept
            N-opin-op:      get the main operator from input.

            N-opsc-opin:    from a concept of the selector to the op-in
            N-opsc-opso:    from a concept of the selector to a relation in the selector
            N-opso-opin:    from a relation of the selector to the op-in
            N-opso-opsc:    from a relation of the selector to a concept in the selector

            N-attr-c:       get the attribute's concept type
            N-attr-parent:  attribute node's parent node
            N-attr-attr:    attribute node's child node as its attribute
            N-attr-child:   attribute node's child node as a operator node
            
            N-opop-op:      op-op's container op.

            N-input-attr:   get the attribute of input node.
            N-input-child:  get the child operator of the input.

            N-concept-child: get the child operator of the constant concept.

            N-coc:   neighbors in the concept and operator representation.
        """
        op_type = get_op_type(op_name)
        neighbors = {}
        if op_type == "op":
            # N-op-parent:
            for op_neighbor in self.parent_operators(op_name):
                record_data(neighbors, [self.get_node_name(op_neighbor)], ["N-op-parent"])
            # N-op-child:
            for op_neighbor in self.child_operators(op_name):
                record_data(neighbors, [self.get_node_name(op_neighbor)], ["N-op-child"])
            # N-opout-c:
            out_node = self.get_to_outnode(op_name)
            concept_type = out_node.split(":")[-1]
            record_data(neighbors, ["c${}".format(concept_type)], ["N-opout-c"])
            # N-op-in:
            in_nodes = self.parent_nodes(op_name)
            record_data(neighbors, in_nodes, ["N-op-in"] * len(in_nodes))

        elif op_type == "op-in":
            # N-opin-c:
            concept_type = op_name.split(":")[-1]
            record_data(neighbors, ["c${}".format(concept_type)], ["N-opin-c"])

            # N-opin-op:
            record_data(neighbors, [self.operator_name(op_name)], ["N-opin-op"])

            # N-opin-opsc/opso:
            selector_dict = self.get_selector(op_name)
            if len(selector_dict) > 0:
                assert len(selector_dict) == 1
                op_in_name = next(iter(selector_dict))
                selector = selector_dict[op_in_name]
                # N-opin-opsc:
                for selector_node in selector.nodes:
                    selector_node_name = op_in_name + "->sc$" + selector_node
                    record_data(neighbors, [selector_node_name], ["N-opin-opsc"])
                # N-opin-opso:
                for edge, re_list in selector.get_relations().items():
                    for re_name in re_list:
                        selector_edge_name = op_in_name + "->so$" + f"({edge[0]},{edge[1]}):{re_name}"
                        record_data(neighbors, [selector_edge_name], ["N-opin-opso"])
        elif op_type in ["op-sc", "op-so"]:
            # op-sc: e.g. "Draw-1->sc$obj_0:c0"
            # op-so: e.g. "Draw-1->so$(obj_0:c0,obj_1:c1):r1"

            # N-opsc/opso-opin:
            op_in, ops = op_name.split("->")
            if ops.startswith("sc$"):
                ops_type = "opsc"
                ops_neighbor_type = "opso"
            elif ops.startswith("so$"):
                ops_type = "opso"
                ops_neighbor_type = "opsc"
            else:
                raise
            record_data(neighbors, [op_in], [f"N-{ops_type}-opin"])

            # N-opsc-opso/N-opso-opsc:
            selector_dict = self.get_selector(op_in)
            selector = selector_dict[next(iter(selector_dict))]
            ops_neighbors = selector.get_neighbors(ops.split("$")[-1])
            for ops_neighbor in ops_neighbors:
                record_data(neighbors, [op_in + f"->{ops_neighbor_type[2:]}$" + ops_neighbor], [f"N-{ops_type}-{ops_neighbor_type}"])
        elif op_type == "op-attr":
            # N-attr-c:
            concept_type = self.get_node_name(op_name).split(":")[-1]
            record_data(neighbors, ["c${}".format(concept_type)], ["N-attr-c"])

            # N-attr-parent:
            attr_parent = self.parent_nodes(op_name)
            assert len(attr_parent) == 1
            attr_parent = attr_parent[0]
            parent_node_type = self.get_node_type(attr_parent)
            record_data(neighbors, [self.get_node_name(self.operator_name(attr_parent))], ["N-attr-parent"])

            # N-attr-child:
            attr_children = self.child_nodes(op_name)
            for attr_child in attr_children:
                child_node_type = self.get_node_type(attr_child)
                if child_node_type == "attr":
                    record_data(neighbors, [attr_child], ["N-attr-attr"])
                else:
                    assert child_node_type == "fun-in"
                    record_data(neighbors, [self.operator_name(attr_child)], ["N-attr-child"])
        elif op_type == "op-op":
            record_data(neighbors, [op_name.split("->")[0]], ["N-opop-op"])
        elif op_type == "input":
            input_children = self.child_nodes(op_name)
            for input_child in input_children:
                input_child_type = self.get_node_type(input_child)
                if input_child_type == "attr":
                    record_data(neighbors, [input_child], ["N-input-attr"])
                else:
                    assert input_child_type == "fun-in"
                    record_data(neighbors, [self.operator_name(input_child)], ["N-input-child"])
        elif op_type == "concept":
            # N-concept-child:
            for op_neighbor in self.child_operators(op_name):
                record_data(neighbors, [self.operator_name(op_neighbor)], ["N-concept-child"])
        elif op_type in ["o", "c", "result", "target", "opparse"]:
            pass
        else:
            raise Exception("op_type '{}' is not supported!".format(op_type))
        return neighbors


    ########################################
    # Operations on PyTorch variables:
    ########################################
    def init_variable(self, keys, init_all=False):
        """Initialize PyTorch variables for dangling nodes."""
        if init_all:
            for variable_name in self.get_variables():
                delattr(variable_name)
        control_nodes = self.control_nodes
        if not hasattr(self, "ground_node_name"):
            self.ground_node_name = DEFAULT_OBJ_TYPE
        for node, is_input in self.input_nodes.items():
            for key in keys:
                variable_name = "variable_{}_{}".format(node, key)
                # is_dangling is True if the node does not expect any input:
                is_dangling = True if is_input and self.get_node_content(node).mode != self.ground_node_name else False
                if is_dangling or node in control_nodes:
                    if not hasattr(self, variable_name):
                        placeholder = self.get_node_content(node)
                        tensor = init_tensor(placeholder)
                        # For now, don't pass gradients through ungrounded variable
                        setattr(self, variable_name, tensor)
#                         setattr(self, variable_name, nn.Parameter(tensor))
                else:
                    if hasattr(self, variable_name):
                        delattr(self, variable_name)


    def get_variables(self, variable_key=None, example_key=None):
        """Get all the PyTorch variables (excluding the PyTorch modules)."""
        variable_dict = OrderedDict()
        for variable_name, tensor in self.named_parameters():
            if variable_name.startswith("variable_"):
                if example_key is not None and variable_key is not None:
                    key_str = "_{}_{}".format(example_key, variable_key)
                elif example_key is not None:
                    key_str = "_{}".format(example_key)
                elif variable_key is not None:
                    key_str = "_{}".format(variable_key)
                else:
                    key_str = None
                if key_str is not None:
                    if key_str in variable_name:
                        variable_dict[variable_name] = tensor
                else:
                    variable_dict[variable_name] = tensor
        return variable_dict


    def get_variables_gen(self, key=None):
        """Get a generator of all the PyTorch variables (excluding the PyTorch modules)."""
        for variable_name, tensor in self.named_parameters():
            if variable_name.startswith("variable_"):
                if key is None:
                    yield tensor
                else:
                    if variable_name.endswith("_{}".format(key)):
                        yield tensor


    def get_targets_from_variables(self):
        """Obtain a collection of targets from ungrounded variables."""
        targets_dict = OrderedDict()
        for node, is_input in self.input_nodes.items():
            if is_input:
                variable_dict = self.get_variables(variable_key=node)
                if len(variable_dict) > 0:
                    concept_name = node.split(":")[-1]
                    targets = OrderedDict([[try_eval(key.split("_")[-1]), CONCEPTS[concept_name].copy().set_node_value(value.detach())] for key, value in variable_dict.items()])
                    targets_dict[node] = targets
        return targets_dict


    def get_fun_modules(self, key=None):
        """Get all the PyTorch modules serving as functions."""
        variable_dict = OrderedDict()
        for variable_name, tensor in self.named_parameters():
            if variable_name.startswith("fun_"):
                if key is not None:
                    if variable_name.endswith("_{}".format(key)):
                        variable_dict[variable_name] = tensor
                else:
                    variable_dict[variable_name] = tensor
        return variable_dict


    def get_fun_modules_gen(self, key=None):
        """Get a generator of all the PyTorch modules serving as functions."""
        for variable_name, tensor in self.named_parameters():
            if variable_name.startswith("fun_"):
                if key is None:
                    yield tensor
                else:
                    if variable_name.endswith("_{}".format(key)):
                        yield tensor


    def set_variable(self, node_name, tensor, example_keys=None):
        """Set variable value at node ${node_name}"""
        variable_name = "variable_{}".format(node_name) if not node_name.startswith("variable_") else node_name
        if not isinstance(tensor, torch.Tensor):
            tensor = to_Variable(tensor, is_cuda=self.is_cuda)
        if example_keys is None:
            if hasattr(self, variable_name):
                getattr(self, variable_name).data = tensor
            else:
                setattr(self, variable_name, nn.Parameter(tensor))
        else:
            for i, key in enumerate(example_keys):
                variable_key = variable_name + "_{}".format(key)
                if hasattr(self, variable_key):
                    getattr(self, variable_key).data = tensor[i]
                else:
                    setattr(self, variable_key, nn.Parameter(tensor[i]))
        return self


    def set_variables(self, variable_dict):
        """Set variable values at given by variable_dict."""
        for variable_name, tensor in variable_dict.items():
            self.set_variable(variable_name, tensor)
        return self


    def set_variable_values(self, node_names, search_variable_values, shapes, example_keys=None):
        """Set multiple variable values at once."""
        start_idx = 0
        for idx, variable in enumerate(node_names):
            size = int(np.prod(shapes[idx]))
            if example_keys is not None:
                length = len(example_keys)
                shape = (length,) + shapes[idx]
            else:
                length = 1
                shape = shapes[idx]
            self.set_variable(variable, 
                              search_variable_values[start_idx: start_idx + size * length].reshape(shape),
                              example_keys=example_keys,
                             )
            start_idx = start_idx + size * length
        return self


    def round_variables(self, key=None):
        """Round all variables whose type is cat or N."""
        for variable_name, variable in self.get_variables().items():
            operator = self.get_node_content(variable_name.split("_")[1]).mode
            dtype = CONCEPTS[operator].get_node_content().mode.dtype
            if key is None:
                if dtype in ["cat", "N"]:
                    self.set_variable("_".join(variable_name.split("_")[1:]), variable.round())
                elif dtype in ["bool"]:
                    self.set_variable("_".join(variable_name.split("_")[1:]), variable.bool())
            else:
                if variable_name.endswith("_{}".format(key)):
                    if dtype in ["cat", "N"]:
                        self.set_variable("_".join(variable_name.split("_")[1:]), variable.round())
                    elif dtype in ["bool"]:
                        self.set_variable("_".join(variable_name.split("_")[1:]), variable.bool())
        return self


    def remove_variable(self, node_name):
        """Remove variable held in node ${node_name}"""
        node_name = self.get_node_name(node_name)
        variable_dict = self.get_variables(variable_key=node_name)
        for variable_name in variable_dict:
            delattr(self, variable_name)
        return self


    def remove_all_variables(self):
        """Remove all PyTorch variables."""
        for variable_name in self.get_variables():
            delattr(self, variable_name)
        return self


    ########################################
    # Operations on the graph structure:
    ########################################
    def connect_nodes(self, node1, node2, type=None):
        """Connect an output node to an arg node."""
        node_source = self.get_node_name(node1)
        node_target = self.get_node_name(node2)
        node_source_content = self.get_node_content(node_source)
        node_target_content = self.get_node_content(node_target)
        if isinstance(node_target_content, Graph):
            # Target node is a criteria node:
            source_mode = node_source.split(":")[-1]
            if type is None:
                id = len(self.parent_nodes(node_target)) + 1
                arg_node_name = "{}-{}:{}".format(node_target, id, source_mode)
                self.add_node(arg_node_name, type="fun-in")
                self.add_edge(node_source, arg_node_name, type="inter-criteria")
                self.add_edge(arg_node_name, node_source, type="b-inter-criteria")
                self.add_edge(arg_node_name, node_target, type="intra")
                self.add_edge(node_target, arg_node_name, type="b-intra")
                self.get_node_content(node_target).init_input_placeholder_nodes({id: source_mode})
            elif type == "Ctrl":
                assert self.get_node_type(node_source) == "fun-in", "Only input nodes can be fed into a control node."
                ctrl_node_name = "{}-0:Ctrl".format(node_target)
                self.add_node(ctrl_node_name, type="fun-in")
                self.add_edge(node_source, ctrl_node_name, type="inter-criteria")
                self.add_edge(ctrl_node_name, node_source, type="b-inter-criteria")
                self.add_edge(ctrl_node_name, node_target, type="intra")
                self.add_edge(node_target, ctrl_node_name, type="b-intra")
            else:
                raise
        else:
            self.add_edge(node_source, node_target, type="inter-input")
            self.add_edge(node_target, node_source, type="b-inter-input")

        if (node_source, node_target) in self.edges:
            # Remove ungrounded variable after connection:
            if len(self.get_variables(variable_key=node_target)) > 0:
                self.remove_variable(node_target)
        return self


    def sort_nodes(self, *nodes):
        """Sort the two operators according to the topological sort."""
        nodes = [self.get_node_name(node) for node in nodes]
        nodes_sorted = [node for node in self.topological_sort if node in nodes]
        return tuple(nodes_sorted)


    def add_connection(self, source_operator, target_operator):
        """Connect two operators."""
        source_operator = self.get_node_name(self.operator_name(source_operator))
        target_operator = self.get_node_name(self.operator_name(target_operator))
        node1 = self.get_to_outnode(source_operator)
        mode = node1.split(":")[-1]
        is_connected = False
        in_nodes = self.get_node_input(target_operator, mode=mode)
        if in_nodes is None:
            return is_connected
        for node2 in in_nodes:
            if len(self.parent_nodes(node2)) > 0 and "multi*" not in node2:
                continue
            node2_mode = canonical(node2.split(":")[-1])
            if node2_mode == mode or Placeholder(node2_mode).accepts(Placeholder(mode)):
                self.connect_nodes(node1, node2)
                is_connected = True
                break
        if self.verbose >= 1 and not is_connected:
            print("Fail to connect {} and {}.".format(source_operator, target_operator))
        return is_connected


    def is_neighbor_op(self, operator1, operator2):
        is_neighbor = False
        parent_ops = self.parent_operators(operator1)
        child_ops = self.child_operators(operator1)
        if self.operator_name(operator2) in parent_ops or self.operator_name(operator2) in child_ops:
            is_neighbor = True
        return is_neighbor


    def break_connection(self, operator1, operator2):
        """Break the connection between two operators."""
        source_node, target_node = self.sort_nodes(operator1, operator2)
        is_connected, node_name_path = self.get_path(source_node, target_node)

        assert is_connected, "There is no path from '{}' to '{}'".format(source_node, target_node)
        node1, node2 = node_name_path[-3:-1]
        assert "inter" in self[node1][node2][0]["type"], "The connection between {} and {} is not an inter-operator edge!".format(node1, node2)
        self.remove_edge(node1, node2)
        self.remove_edge(node2, node1)
        return self


    @property
    def isolated_operators(self):
        strongly_connected_components = sorted(nx.strongly_connected_components(self), key=len)
        strongly_connected_operators = [list({node.split("-")[0] for node in strongly_connected_component}) for strongly_connected_component in strongly_connected_components]
        operators_isolated = []
        if len(strongly_connected_operators) > 1:
            for operator_list in strongly_connected_operators:
                if len(operator_list) == 1 and "input" not in operator_list[0]:
                    operators_isolated += operator_list
        return operators_isolated


    def init_input_placeholder_nodes(self, node_dict=None):
        """Initialize input_place_holder_nodes, where the node_dict={i: mode} are input-{i}:{mode} pairs."""
        node_dict = {1: self.ground_node_name if self.ground_node_name is not None else DEFAULT_OBJ_TYPE} if node_dict is None else node_dict
        for i, mode in node_dict.items():
            try:
                self.get_node_name("input-{}".format(i))
            except:
                self.add_node("input-{}:{}".format(i, mode),
                              value=Placeholder(mode), type="input")
        return self


    def connect_input_placeholder_nodes(self, node_dict=None):
        """Feed each ground node to the all the ungrounded input nodes that matches."""
#         if len(self.input_placeholder_nodes) == 0:
        self.init_input_placeholder_nodes(node_dict=node_dict)
        input_placeholder_node_connected = []
        for input_node, is_input in self.input_nodes.items():
            if is_input and len(self.get_variables(variable_key=input_node)) == 0:
                content = self.nodes(data=True)[input_node]
                for i, node in enumerate(self.input_placeholder_nodes):
                    if Placeholder(input_node.split(":")[-1]).accepts(Placeholder(node.split(":")[-1])):
                        if node not in input_placeholder_node_connected:
                            self.connect_nodes(node, input_node)
                            input_placeholder_node_connected.append(node)
                            break
                        else:
                            if i == len(self.input_placeholder_nodes) - 1:
                                self.connect_nodes(node, input_node)
                                break
        return self


    def add_get_attr(self, node_name, attr_name):
        """Add an operation that obtains the attribute of the output at 'node_name'."""
        node_name = self.get_node_name(node_name)
        assert self.get_node_type(node_name) in ["fun-out", "input", "attr"], "Only output node, attribute node and input-placeholder can get attributes!"
        concepts = combine_dicts([CONCEPTS, NEW_CONCEPTS])
        concept = concepts[node_name.split(":")[-1]]
        is_connected = False
        for concept_attr_name in concept.attributes:
            if concept_attr_name.split(":")[0] == attr_name:
                is_connected = True
                break
        if is_connected:
            attr_name = concept.get_node_name(attr_name)
            attr_mode = attr_name.split(":")[-1]
            output_name = "{}^{}".format(node_name.split(":")[0], attr_name)
            self.add_node(output_name, value=Placeholder(attr_mode), type="attr")
            self.add_edge(node_name, output_name, type="intra-attr")
            self.add_edge(output_name, node_name, type="b-intra-attr")
        return is_connected


    def remove_get_attr(self, node_name, attr_node_name):
        """Remove an operation that obtains the attribute of the output at 'node_name'"""
        node_name = self.get_node_name(node_name)
        attr_node_name = self.get_node_name(attr_node_name)
        assert self.get_node_type(node_name) in ["fun-out", "input", "attr"], "Only output node, attribute node and input-placeholder can get attributes!"
        assert attr_node_name in self.child_nodes(node_name), "the attribute node {} must be the child_node of the {}".format(attr_node_name, node_name)
        path = self.get_path_to_output(attr_node_name)
        self.remove_nodes_from(path)
        return self


    def propose_recruit(self, operator_name, includes=["input", "output"]):
        """Find all the possible operators that can connect to the operator in the current Graph."""
        recruit_candidates = {}

        if "output" in includes:
            # Get output recruit candidates:
            output_node = self.child_nodes(operator_name)
            assert len(output_node) == 1
            output_node = output_node[0]
            assert self.get_node_type(output_node) in ["fun-out", "attr"]
            if len(self.child_nodes(output_node)) == 0:
                output_mode = output_node.split(":")[-1]
                recruit_candidates[output_node]= INPUT_MODE_DICT[output_mode]

        if "input" in includes:
            # Get input recruit candidates:
            input_nodes = self.parent_nodes(operator_name)
            input_nodes_global = self.input_nodes
            for input_node in input_nodes:
                assert self.get_node_type(input_node) == "fun-in"
                if input_nodes_global[input_node] is True:
                    input_mode = input_node.split(":")[-1]
                    recruit_candidates[input_node] = OUTPUT_MODE_DICT[input_mode]
        return recruit_candidates


    def propose_recruit_all(self):
        """Find all the possible ways the current Graph can recruit a new operator."""
        recruit_cand_dict = {}
        for operator_name in self.operators:
            if operator_name not in self.goals:
                recruit_cand_dict[operator_name] = self.propose_recruit(
                    operator_name, includes=["input", "output"])
        return recruit_cand_dict


    def get_subgraph(self, operator_names, includes_constant=True):
        """Obtain the subgraph corresponding to the operator_names.
        
        Args:
            operator_names: list of strings indicating the operators to be included in the subgraph.
        """
        if isinstance(operator_names, str):
            operator_names = [operator_names]
        assert isinstance(operator_names, list)
        # Collect the nodes corresponding to operator_names:
        nodes = []
        for operator_name in operator_names:
            assert operator_name in self.operators, "Operator {} is not in self.operators!".format(operator_name)
            for node in self.nodes:
                if node.startswith("{}".format(operator_name)):
                    nodes.append(node)
                    if includes_constant:
                        for parent_node in self.parent_nodes(node):
                            if self.get_node_type(parent_node) == "concept":
                                nodes.append(parent_node)
        # Obtain subgraph:
        subgraph = Graph(self.subgraph(nodes))
        # Set up operators, goals and ground_node_name:
        subgraph.operators = operator_names
        subgraph.name = subgraph.operators_core[0]
        subgraph.goals = [operator_name for operator_name in self.goals if operator_name in operator_names]
        subgraph.funs = [operator_name for operator_name in self.funs if operator_name in operator_names]
        subgraph.ground_node_name = self.ground_node_name
        # Set up PyTorch variables for subgraph:
        subgraph.remove_all_variables()
        for variable_name, tensor in self.get_variables().items():
            for operator_name in operator_names:
                if operator_name in variable_name:
                    setattr(subgraph, variable_name, nn.Parameter(tensor))
        # Set up Pytorch Modules:
        for fun_name in self.funs:
            for operator_name in operator_names:
                if fun_name in operator_names:
                    setattr(subgraph, "fun_{}".format(fun_name), getattr(self, "fun_{}".format(fun_name)))
        return subgraph


    def add_subgraphs(self, subgraphs, **kwargs):
        """Add multiple subgraphs into the current graph."""
        for subgraph in subgraphs:
            self.add_subgraph(subgraph, **kwargs)
        return self


    def add_subgraph(
        self,
        subgraph,
        is_tentative=False,
        **kwargs
    ):
        """Add subgraph. If failing to connect the current graph with the subgraph, revert to original graph."""
        if not is_tentative:
            self.add_subgraph_core(subgraph, **kwargs)
            return self
        else:
            g = deepcopy(self)
            is_connected = g.add_subgraph_core(subgraph, **kwargs)
            if not is_connected:
                return self
            else:
                # Replace self by g:
                self.__dict__.update(g.__dict__)
                return self


    def add_subgraph_core(
        self,
        subgraph,
        is_constant=False,
        add_full_concept=True,
        rename_conflict=True,
        is_share_fun=False,
        is_connect=False,
        is_remove_input_placeholder=True,
    ):
        """Add a new subgraph into the current graph.

        Args:
            subgraph: a Graph() object to be added to the current graph, or a Concept() object to act as input or constant.
            is_constant: (only for concept subgraph) if True, the concept added will become a concept node, otherwise it will be an input node.
            add_full_concept: (only for concept subgraph) if True, all the properties of the concept will be expanded from the input node.
            rename_conflict: (only for operator subgraph) if True, rename conflicting operators in subgraph.
                If False, conflicting operators in subgraph will overwrite the ones in self.
            is_share_fun: (only for operator subgraph) if True, operators with conflicting names will still share the PyTorch modules
            is_connect: if True, will connect the current output of self to the input of newly added subgraph.
        """
        # If it is a Concept node:
        if isinstance(subgraph, Concept):
            if is_constant:
                self.last_added_name = "concept-{}:{}".format(len(self.constant_concept_nodes) + 1, subgraph.name)
                self.add_node(self.last_added_name, value=subgraph.copy(), type="concept")
            else:
                if add_full_concept in [True, "basic"]:
                    base_name = "input-{}".format(len(self.input_placeholder_nodes) + 1)
                    self.last_added_name = input_name = "{}:{}".format(base_name, subgraph.name)
                    self.add_node(input_name, type="input", value=Placeholder(subgraph.name), fun=subgraph.get_node_fun())
                    if add_full_concept is True:
                        nodes_to_add = list(subgraph.nodes)
                    elif add_full_concept == "basic":
                        nodes_to_add = [subgraph.name]
                    for node in nodes_to_add:
                        if node == subgraph.name:
                            node1 = "{}:{}".format(base_name, subgraph.name)
                        else:
                            node1 = "{}^{}".format(base_name, node)
                        for adj, item in subgraph[node].items():
                            if "b" not in item[0]["type"]:
                                node2 = "{}^{}".format(base_name, adj)
                                if node1 != input_name:
                                    self.add_node(node1, type="attr", value=Placeholder(node1.split(":")[-1]),
                                                  fun=subgraph.get_node_fun(node))
                                if node2 != input_name:
                                    self.add_node(node2, type="attr" if node2 != input_name else "input",
                                                  value=Placeholder(adj.split(":")[-1]),
                                                  fun=subgraph.get_node_fun(adj))
                                self.add_edge(node1, node2, type="intra-attr")
                                self.add_edge(node2, node1, type="b-intra-attr")
                else:
                    self.last_added_name = "input-{}:{}".format(len(self.input_placeholder_nodes) + 1, subgraph.name)
                    self.add_node(self.last_added_name, value=Placeholder(subgraph.name), type="input")
            return self

        # Rename the operators that have name conflict:
        self.rename_mapping = None
        if len(self.operators) > 0 and rename_conflict:
            mapping = get_rename_mapping(self.operators, subgraph.operators)
            if len(mapping) > 0:
                subgraph = subgraph.copy(is_share_fun=is_share_fun).rename_operators(mapping)
                self.rename_mapping = mapping
        
        if is_connect:
            output_node = self.output_nodes
            if len(output_node) > 0:
                output_node = output_node[-1]
            else:
                output_node = None

        # Add operator:
        self.last_added_name = subgraph.name
        operator_copy = deepcopy(subgraph)
        if is_remove_input_placeholder:
            operator_copy.remove_input_placeholder_nodes()
        self.add_nodes_from(operator_copy.nodes(data=True))
        self.add_edges_from(operator_copy.edges(data=True))
        assert nx.is_directed_acyclic_graph(self.forward_graph(is_copy_module=False))

        # Add variables:
        for variable_name, tensor in subgraph.get_variables().items():
            setattr(self, variable_name, nn.Parameter(tensor))

        # Add funs:
        self.funs = list(set(self.funs).union(subgraph.funs))
        for fun_name in subgraph.funs:
            setattr(self, "fun_{}".format(fun_name), getattr(subgraph, "fun_{}".format(fun_name)))
            self.set_node_content(getattr(self, "fun_{}".format(fun_name)), fun_name)

        # Combine self.operators:
        self.operators = remove_duplicates(self.operators + subgraph.operators)

        # Combine self.goals:
        self.goals = remove_duplicates(self.goals + subgraph.goals)

        # Update self.ground_node_name:
        if subgraph.ground_node_name is not None:
            if self.ground_node_name is None:
                self.ground_node_name = subgraph.ground_node_name

        # Connect the current output_node with the newly added subgraph:
        if is_connect and output_node is not None:
            is_connected = self.add_connection(self.operator_name(output_node), self.last_added_name)
            return is_connected
        else:
            return False


    def compose(self, subgraph):
        """Compose a new subgraph into the current graph, where node and edge with the same name are unified.

        Args:
            subgraph: a Graph() object to be added to the current graph.
        """
        # If it is a Concept node:
        assert isinstance(subgraph, Graph)

        # Add operator:
        subgraph = subgraph.copy()
        G = nx.compose(self, subgraph)
        assert nx.is_directed_acyclic_graph(G.forward_graph(is_copy_module=False))

        # Add variables:
        this_variables = self.get_variables()
        for variable_name, tensor in subgraph.get_variables().items():
            if variable_name not in this_variables:
                setattr(G, variable_name, nn.Parameter(tensor))

        # Add funs:
        G.funs = remove_duplicates(self.funs + subgraph.funs)
        for fun_name in subgraph.funs:
            setattr(G, "fun_{}".format(fun_name), getattr(subgraph, "fun_{}".format(fun_name)))

        # Combine self.operators:
        G.operators = remove_duplicates(self.operators + subgraph.operators)

        # Combine self.goals:
        G.goals = remove_duplicates(self.goals + subgraph.goals)

        # Update self.ground_node_name:
        if subgraph.ground_node_name is not None:
            if self.ground_node_name is None:
                G.ground_node_name = subgraph.ground_node_name
        self.__dict__.update(G.__dict__)
        return self
    
     
    def incorporate_goal_subgraph(self):
        """Build a new computation graph that incorporates all goal subgraphs."""
        G = self.copy()
        goal_output_nodes = []
        goal_ctrl_nodes = []
        for goal_name in self.goals_optm:
            goal_output_nodes.append(G.child_nodes(goal_name)[0])
            goal_input_nodes = [node for node in G.parent_nodes(goal_name) if not node.endswith("-0:Ctrl")]
            arg_nodes = [G.parent_nodes(node)[0] for node in goal_input_nodes]
            goal_ctrl = "{}-0:Ctrl".format(goal_name)
            ctrl_nodes = G.parent_nodes(goal_ctrl)
            goal_ctrl_nodes = goal_ctrl_nodes + ctrl_nodes
            for arg_node, goal_input_node in zip(arg_nodes, goal_input_nodes):
                G.set_edge_type(arg_node, goal_input_node, "inter-input")
            for ctrl_node in ctrl_nodes:
                G.remove_edge(ctrl_node, goal_ctrl)
            G.remove_node(goal_ctrl)
        G.goals = []
        return G, (goal_ctrl_nodes, goal_output_nodes)


    def remove_subgraph(self, subgraph, is_rename=True):
        """Delete the subgraph, including its args and output, from current graph.

        Args:
            subgraph: a str indicating the name of operator to delete, or a list of strings indicating
                the list of names of operators to delete, or a Graph() object to delte.
        """
        nodes_to_remove = []
        if isinstance(subgraph, str):
            operator_names = [subgraph]
        elif isinstance(subgraph, list):
            operator_names = subgraph
        else:
            operator_names = deepcopy(subgraph.operators)

        # Remove nodes:
        for node in self.nodes:
            for operator_name in operator_names:
                if node.startswith("{}".format(operator_name)):
                    if node.startswith("{}-".format(operator_name)):
                        nodes_to_remove.append(node)
                    else:
                        if node == operator_name:
                            nodes_to_remove.append(node)
        self.remove_nodes_from(nodes_to_remove)

        # Remove operators from self.operators and self.goals:
        for operator_name in operator_names:
            self.operators.remove(operator_name)
            if operator_name in self.goals:
                self.goals.remove(operator_name)

        # Remove PyTorch variables:
        for variable_name in self.get_variables():
            for operator_name in operator_names:
                if "variable_{}-".format(operator_name) in variable_name:
                    delattr(self, variable_name)

        # Remove Pytorch funs:
        funs_to_remove = []
        for fun_name in self.funs:
            if fun_name in operator_names:
                funs_to_remove.append(fun_name)
                delattr(self, "fun_{}".format(fun_name))
        self.funs = list(set(self.funs).difference(set(funs_to_remove)))
        
        # Canonicalize operators:
        if is_rename:
            mapping = canonicalize_strings(self.operators)
            self.rename_operators(mapping)
        return self


    def remove_input_placeholder_nodes(self):
        """Remove input_placeholder_nodes."""
        nodes_to_remove = self.input_placeholder_nodes
        self.remove_nodes_from(nodes_to_remove)
        return self


    def preserve_subgraph(self, output_nodes, level="node"):
        """Preserve the minimal subgraph such that all the nodes in output_nodes can be calculated."""
        if not isinstance(output_nodes, list):
            output_nodes = [output_nodes]
        nodes_required = self.get_ancestors(output_nodes, includes_self=True)
        if level in ["operator", "node"]:
            operators_to_preserve = remove_duplicates([node.split("-")[0] for node in nodes_required])
            operators_to_remove = [operator for operator in self.operators if operator not in operators_to_preserve]
            self.remove_subgraph(operators_to_remove)
        if level == "node":  # Remove all unused nodes
            nodes_to_remove = [node for node in self.nodes if node not in nodes_required]
            for node in nodes_to_remove:
                if node in self.nodes:
                    self.remove_node(node)
        return self


    def rename_operators(self, mapping):
        """Rename operators based on the dictionary of mapping."""
        # Expand mapping to include all relevant nodes:
        mapping_expanded = {}
        for node_name in self.nodes:
            for operator_name, new_operator_name in mapping.items():
                if node_name.startswith(operator_name):
                    mapping_expanded[node_name] = new_operator_name + node_name[len(operator_name):]

        # Build a new graph:
        G = nx.relabel_nodes(self, mapping_expanded)

        # Build other properties:
        G.operators = [mapping[c] if c in mapping else c for c in self.operators]
        G.goals = [mapping[c] if c in mapping else c for c in self.goals]
        G.funs = [mapping[c] if c in mapping else c for c in self.funs]
        G.ground_node_name = mapping[self.ground_node_name] if self.ground_node_name in mapping else self.ground_node_name
        if G.name in mapping:
            G.name = mapping[G.name]
        for variable_name, variable in self.get_variables().items():
            variable_name_new = "variable_" + mapping_expanded[variable_name[9:]] if variable_name[9:] in mapping_expanded else variable_name
            setattr(G, variable_name_new, nn.Parameter(variable))
        for fun_name in self.funs:
            fun_name_new = mapping_expanded[fun_name] if fun_name in mapping_expanded else fun_name
            setattr(G, "fun_" + fun_name_new, getattr(self, "fun_{}".format(fun_name)))
            G.set_node_content(getattr(G, "fun_" + fun_name_new), fun_name_new)
        self.__dict__.update(G.__dict__)
        return self


    def add_And_over_bool(self):
        """If the number of Boolean output_node is greater than 1, combine them using And."""
        bool_output_nodes = []
        for output_node in self.get_output_nodes(types=["fun-out", "attr", "input"], dangling_mode=True):
            mode = output_node.split(":")[-1]
            if mode == "Bool":
                bool_output_nodes.append(output_node)
        if len(bool_output_nodes) > 1:
            self.add_subgraph(OPERATORS["And"])
            for output_node in bool_output_nodes:
                self.connect_nodes(output_node, "And-1")
            bool_output_node = "And-o:Bool"
        elif len(bool_output_nodes) == 1:
            bool_output_node = bool_output_nodes[0]
        else:
            bool_output_node = None
        return bool_output_node


    def add_Or_over_bool(self):
        """If the number of Boolean output_node is greater than 1, combine them using Or."""
        bool_output_nodes = []
        for output_node in self.get_output_nodes(types=["fun-out", "attr", "input"], dangling_mode=True):
            mode = output_node.split(":")[-1]
            if mode == "Bool":
                bool_output_nodes.append(output_node)
        if len(bool_output_nodes) > 1:
            self.add_subgraph(OPERATORS["Or"])
        for output_node in bool_output_nodes:
            self.connect_nodes(output_node, "Or-1")
        return self


    def abstract(self, node_dict=None):
        """Turn the graph into a operator node."""
        g = self.copy()
        g.connect_input_placeholder_nodes(node_dict=node_dict)
        args = [Placeholder(node_name.split(":")[-1]) for node_name in g.input_placeholder_nodes]
        output_nodes = self.get_output_nodes(types=["attr", "fun-out"], dangling_mode=True)
        for node_name in reversed(self.topological_sort):
            if node_name in output_nodes:
                break
        output_mode = node_name.split(":")[-1]
        G = Graph(name=self.name,
                  repr=to_Variable(torch.rand(REPR_DIM), is_cuda=self.is_cuda),
                  forward={"args": args,
                           "output": Placeholder(output_mode),
                           "fun": g,
                          })
        outer_node_dict = OrderedDict()
        for i, arg in enumerate(args):
            outer_node_dict[i + 1] = arg.mode
        G.remove_input_placeholder_nodes()
        return G


    def to(self, device):
        super(Graph, self).to(device)
        selector_dict = self.get_selectors()
        for key in selector_dict:
            selector_dict[key].to(device)
        return self


    @property
    def DL(self):
        """Recursively computing DL."""
        DL = len(self.operators_full) + self.edge_index.shape[1]
        for op_name in self.operators_core:
            op = self.get_node_content(op_name)
            if isinstance(op, BaseGraph):
                DL += op.DL
        return DL


class Do(BaseGraph):
    """A selector, that selects one of the operator to run. If selector is not given,
    output results from all operators.
    """
    def __init__(self, G=None, **kwargs):
        self.operators = []
        super(Do, self).__init__(G=G, **kwargs)
        self.name = "Do"


    def forward(self, selector, *inputs, **kwargs):
        """Forward function. 
        
        Args:
            selector: A categorical PyTorch Tensor. If selector is None, output all possible outputs.
            inputs: the same format as the inputs in the forward() in Graph().
            
        Returns:
            resuls: the same format as the results in the forward() in Graph().
        """
        if isinstance(selector, Concept):
            selector = selector.get_root_value()
        if selector is not None:   
            selector = to_np_array(selector.long())
            assert selector < len(self.operators)

            operator_chosen = self.operators[selector]
            results = operator_chosen(*inputs, **kwargs)
        else:
            results = OrderedDict()
            for i, operator in enumerate(self.operators):
                result = operator(*inputs, **kwargs)
                if isinstance(result, dict):
                    for key, item in result.items():
                        if not isinstance(key, tuple):
                            key = (key,)
                        results[key + ("Do-{}-{}".format(i, operator.name),)] = item
                else:
                    results["Do-{}-{}".format(i, operator.name)] = result
        return results


    def add_subgraph(self, subgraph):
        """Add operator to self's self.operators collection."""
        for operator in self.operators:
            assert subgraph.name != operator.name
        self.last_added_name = subgraph.name
        self.operators.append(subgraph)


    def __str__(self):
        repr_str = self.graph["name"] if "name" in self.graph else "Graph"
        # Composing content string:
        content_str = ""
        operator_list = []
        for operator in self.operators:
            operator_list.append(operator.name)
            operator.draw()
        if len(operator_list) > 0:
            content_str += "operators={}, ".format(operator_list)
        return '{}({})'.format(repr_str, content_str[:-2])


    def __repr__(self):
        return self.__str__()


    def draw(self):
        """Draw the operator components."""
        if len(self.operators) > 0:
            for operator in self.operators:
                print("{}:".format(operator.name))
                operator.draw()


    @property
    def DL(self):
        return len(self.operators) + 1  # 1 is for input node


# In[ ]:


class ForGraph(Graph):
    """Perform While Loop on the provided G as actioin and criteria is satisfied, break the loop."""
    def __init__(self, G=None, criteria=None):
        super(ForGraph, self).__init__(G=G)
        self.criteria = criteria
        self.MAX_ITER = 100

    def forward(self, *inputs):
        inputs = list(deepcopy(inputs))
        outputs = {}
        for i in range(self.MAX_ITER):
            output_cand = super(ForGraph, self).forward(*inputs)
            inputs[0] = output_cand
            if self.criteria(*inputs):
                break
            else:
                output = output_cand
                outputs["obj_{}:Image".format(i)] = output
        output_comp, pos_bounding = get_comp_obj(outputs, CONCEPTS)
        output_comp.set_node_value(pos_bounding, "pos")
        return output_comp


# ### 1.3.2 Execute action that edit the graph:

# In[ ]:


def execute_action(
    graph,
    action,
    OPERATORS,
    CONCEPTS,
    max_ops=30,
    allowed_modes="01234",
    verbose=False,
):
    """
    Action space:
        A0: [2 * N + 1]: which graph it is modifying, 0 for the outmost operator graph. (2*0+2, 2*1+2, ... , 2*(N-1)+2) for the 
            operator graph inside the operator i (as sorted by topological_sort), and (2*0+1, 2*1+1, ... , 2*(N-1)+1) for the 
            selector inside the operator i.
        A5 [2]: whether to stop. 0. continue; 1. stop.
        if A0 refers to an operator graph:
            A1: [N]: source operator
            A2: [|Op| + 4 + N + |A|]: target operator
            A3: relation type (here does not matter)
            A4: add/delete edge or change operator. 
                0. add edge from A1 to A2
                1. add edge from A2 to A1
                2. delete edge between A1 and A2
                4. change the operator of A1 to A2
        if A0 refers to a selector:
            A1: [N]: source object
            A2: [|Op| + 4 + N]: target object
            A3: relation type
            A4: add/delete edge or change operator.
                0. add relation from A1 to A2
                1. add relation from A2 to A1
                2. delete relation from A1 to A2
                3. delete relation from A2 to A1

    OPERATORS: here the OPERATORS can be relation or manipulation operators, or constant concept.
    max_ops: maximum number of operators in terms of operators_full.
    allow_modes: a string indicating the allowed action in A3. Use subset of "01234".
    """
    assert len(action) == 6, "action must be multi-discrete with 6 elements!"
    has_effect = False
    len_OPERATORS = len(OPERATORS) + 4
    OPERATOR_KEYS = list(OPERATORS.keys())
    if action[5] == 1:
        # stop, do nothing:
        if verbose >= 1:
            print("**Stop.\n")
        return deepcopy(graph), has_effect
    elif action[5] == 0:
        # do something:
        ops_global = graph.operators_full
        ops_global_reverse = {id: op for op, id in ops_global.items()}
        if action[0] % 2 == 0:
            if action[0] == 0:
                # Recursion level: outmost
                g = deepcopy(graph)
                if verbose >= 1:
                    print("**Operate on the global graph:")
            else:
                # Recursion level: at the inner operator graph in operator with op_id=(action[0] - 2) // 2:
                graph_id = (action[0] - 2) // 2
                graph_node_name = ops_global_reverse[graph_id]
                g = deepcopy(graph.get_node_content(graph_node_name))
                if verbose >= 1:
                    print("**Operate on the component graph of op '{}':".format(graph_node_name))

            ops = g.operators_full
            ops_reverse = {id: op for op, id in ops.items()}
            source_op_id = int(action[1])
            target_op_id = int(action[2])

            if source_op_id < len(ops):
                source_op = ops_reverse[source_op_id]
                if target_op_id < len_OPERATORS:
                    # Target operator is new from OPERATORS:
                    if target_op_id < len(OPERATORS):
                        target_op = OPERATORS[OPERATOR_KEYS[target_op_id]]
#                     elif target_op_id == len(OPERATORS) + 2:
#                         target_op = Graph(
#                             name="Do",
#                             forward={
#                                 "args": [Placeholder(DEFAULT_OBJ_TYPE)],
#                                 "output": Placeholder(DEFAULT_OBJ_TYPE),
#                                 "fun": Do(),
#                             })
                    else:
                        return graph, has_effect

                    if action_equal(action[4], 0, allowed_modes) or action_equal(action[4], 1, allowed_modes):
                        if len(g.operators_full) >= max_ops:
                            if verbose >= 1:
                                print("**Number of operators exceeds max_op={}. Stop.".format(max_ops))
                            return graph, has_effect

                        # Add an edge from A1 to A2 (or reverse):
                        g.add_subgraph(target_op)
                        if action_equal(action[4], 0, allowed_modes):
                            is_connected = g.add_connection(source_op, g.last_added_name)
                            if verbose >= 1:
                                print("\t**Connect the output of op '{}' to a new op '{}'".format(source_op, g.last_added_name))
                        elif action_equal(action[4], 1, allowed_modes):
                            is_connected = g.add_connection(g.last_added_name, source_op)
                            if verbose >= 1:
                                print("\t**Add an op '{}' and connect its output to the input of op '{}'".format(g.last_added_name, source_op))
                        if is_connected:
                            has_effect = True
                        else:
                            # Revert to previous graph if cannot connect:
                            g = deepcopy(graph)
                            if verbose >= 1:
                                print("\t\t**Fail to connect the newly added op. Revert back.")
                    elif action_equal(action[4], 4, allowed_modes):
                        # Change node:
                        # Input nodes are not allowed to change, and not switch node with the same operator type.
                        if "input" not in source_op and split_string(source_op)[0] != target_op.name:
                            if "^" in source_op:
                                # source_op is attribute node:
                                parent_operator = g.parent_operators(source_op)[0]
                                g.remove_get_attr(g.get_to_outnode(parent_operator), source_op)
                                g.add_subgraph(target_op)
                                is_connect = g.add_connection(parent_operator, g.last_added_name)
                                if verbose >= 1:
                                    print("\t**Remove attribute '{}' and replace with op '{}'.".format(source_op, g.last_added_name))
                                if is_connect:
                                    has_effect = True
                                else:
                                    g = deepcopy(graph)
                                    if verbose >= 1:
                                        print("\t\t**Fail to connect the newly added op. Revert back.")
                            else:
                                parent_operators = g.parent_operators(source_op)
                                child_operators = g.child_operators(source_op)
                                g.remove_subgraph(source_op)
                                g.add_subgraph(target_op)
                                is_connect_parent = False
                                is_connect_child = False
                                for parent_op in parent_operators:
                                    is_connect_parent = is_connect_parent or g.add_connection(parent_op, g.last_added_name)
                                for child_op in child_operators:
                                    if "^" in child_op:
                                        # Has attribute nodes:
                                        # Still need to fix the case where there are multiple get_attr nodes and more depth.
                                        g.add_get_attr(g.get_to_outnode(g.last_added_name), child_op.split("^")[-1])
                                        is_connect_child = True
                                    else:
                                        is_connect_child = is_connect_child or g.add_connection(g.last_added_name, child_op)
                                if verbose >= 1:
                                    print("\t**Replace '{}' with '{}'.".format(source_op, target_op.name))
                                if is_connect_parent or is_connect_child:
                                    has_effect = True
                                else:
                                    g = deepcopy(graph)
                                    if verbose >= 1:
                                        print("\t\t**Fail to connect with replaced op. Revert back.")

                elif len_OPERATORS <= target_op_id < len_OPERATORS + len(ops):
                    # Target operator is within current graph:
                    target_op = ops_reverse[target_op_id - len_OPERATORS]
                    if source_op != target_op:
                        if action_equal(action[4], 0, allowed_modes) or action_equal(action[4], 1, allowed_modes):
                            if action_equal(action[4], 0, allowed_modes):
                                is_connect = g.add_connection(source_op, target_op)
                                if verbose >= 1:
                                    print("\t**Add an edge from '{}' to '{}'".format(source_op, target_op))
                            elif action_equal(action[4], 1, allowed_modes):
                                is_connect = g.add_connection(target_op, source_op)
                                if verbose >= 1:
                                    print("\t**Add an edge from '{}' to '{}'".format(target_op, source_op))
                            has_effect = is_connect
                            if not has_effect:
                                if verbose >= 1:
                                    print("\t\t**Cannot add edge.")
                        elif action_equal(action[4], 2, allowed_modes):
                            # Delete an edge:
                            is_neighbor = g.is_neighbor_op(source_op, target_op)
                            if is_neighbor:
                                if "^" in target_op:
                                    g.remove_get_attr(source_op, target_op)
                                    has_effect = True
                                    if verbose >= 1:
                                        print("\t**Remove the attribute '{}' from '{}'.".format(target_op, source_op))
                                else:
                                    g.break_connection(source_op, target_op)
                                    # Remove the target_op if it only has one operator:
                                    g.remove_subgraph(g.isolated_operators)
                                    has_effect = True
                                    if verbose >= 1:
                                        print("\t**Remove an edge between '{}' and '{}'.".format(source_op, target_op))
                            else:
                                if verbose >= 1:
                                    print("\t\t**'{}' and '{}' are not connected. Cannot delete edge.".format(source_op, target_op))

                elif target_op_id >= len(ops) + len_OPERATORS:
                    if action_equal(action[4], 0, allowed_modes):
                        # Add an attribute:
                        source_op_outnode = g.get_to_outnode(source_op)
                        concept_name = source_op_outnode.split(":")[-1]
                        attributes = CONCEPTS[concept_name].attributes
                        k = target_op_id - (len(ops) + len_OPERATORS)
                        if verbose >= 1:
                            print("\t**Add an attribute '{}' on '{}'".format(concept_name, source_op))
                        if k < len(attributes):
                            g.add_get_attr(source_op_outnode, attributes[k])
                            has_effect = True
                        else:
                            if verbose >= 1:
                                print("\t\t**The attribute id exceeds the number of attributes for '{}'".format(source_op))
            # use the g:
            if action[0] == 0:
                graph = g
            else:
                graph.set_node_content(g, graph_node_name)
            if has_effect:
                if verbose >= 2:
                    print("Plotting the operator graph after the action:")
                    graph.draw()
                    print("=" * 100 + "\n")

        elif action[0] % 2 == 1:
            # Recursion level: at the selector of operator with op_id=(action[0] - 2) // 2:
            graph_id = (action[0] - 1) // 2
            graph_node_name = ops_global_reverse[graph_id]
            selector_dict = deepcopy(graph.get_selector(graph_node_name))
            g = selector_dict[next(iter(selector_dict))]
            print("**Operate on '{}''s selector:".format(graph_node_name))

            ops = g.get_graph("obj").topological_sort
            ops_reverse = {id: op for id, op in enumerate(ops)}
            source_op_id = int(action[1])
            target_op_id = int(action[2])

            if source_op_id < len(ops):
                source_op = ops_reverse[source_op_id]
                relation_id = action[3]
                relation_name = OPERATOR_KEYS[relation_id]
                if target_op_id < len_OPERATORS:
                    if action_equal(action[4], 0, allowed_modes) or action_equal(action[4], 1, allowed_modes):
                        if len(ops) >= max_ops:
                            if verbose >= 1:
                                print("**Number of operators exceeds max_op={}. Stop.".format(max_ops))
                            return graph, has_effect

                        # Add an edge from A1 to A2 (or reverse):
                        obj_name = g.add_obj(CONCEPTS[DEFAULT_OBJ_TYPE].copy(), change_root=False, add_full_concept=False)
                        if action_equal(action[4], 0, allowed_modes):
                            g.add_relation_manual(relation_name, source_op, obj_name)
                            if verbose >= 1:
                                print("\t**Add a relation '{}' from '{}' to a new '{}'.".format(relation_name, source_op, obj_name))
                        elif action_equal(action[4], 1, allowed_modes):
                            g.add_relation_manual(relation_name, obj_name, source_op)
                            if verbose >= 1:
                                print("\t**Add a relation '{}' from a new '{}' to '{}'.".format(relation_name, obj_name, source_op))
                        has_effect = True
                elif len_OPERATORS <= target_op_id < len_OPERATORS + len(ops):
                    target_op = ops_reverse[target_op_id - len_OPERATORS]
                    if source_op != target_op:
                        if action_equal(action[4], 0, allowed_modes) or action_equal(action[4], 1, allowed_modes):
                            # Add a specific relation:
                            if action_equal(action[4], 0, allowed_modes):
                                g.add_relation_manual(relation_name, source_op, target_op)
                                if verbose >= 1:
                                    print("\t**Add a relation '{}' from '{}' to '{}'.".format(relation_name, source_op, target_op))
                            elif action_equal(action[4], 1, allowed_modes):
                                g.add_relation_manual(relation_name, target_op, source_op)
                                if verbose >= 1:
                                    print("\t**Add a relation '{}' from '{}' to '{}'.".format(relation_name, target_op, source_op))
                            has_effect = True
                        elif action_equal(action[4], 2, allowed_modes) or action_equal(action[4], 3, allowed_modes):
                            # Delete a specific relation:
                            if action_equal(action[4], 2, allowed_modes):
                                has_effect = g.remove_relation_manual(source_op, target_op, relation_name)
                                if verbose >= 1:
                                    print("\t**Delele the relation '{}' from '{}' to '{}'".format(relation_name, source_op, target_op))
                            elif action_equal(action[4], 3, allowed_modes):
                                has_effect = g.remove_relation_manual(target_op, source_op, relation_name)
                                if verbose >= 1:
                                    print("\t**Delele the relation '{}' from '{}' to '{}'".format(relation_name, target_op, source_op))
            if has_effect:
                graph.set_selector(g, graph_node_name)
                if verbose >= 2:
                    print("Plotting the operator graph after the action:")
                    graph.draw()
                    print("=" * 100 + "\n")
        
        if verbose >= 1:
            print()
        return graph, has_effect


# ## 1.4 Concept:

# ### 1.4.1 Concept

# In[ ]:


class Concept(BaseGraph):
    """Implement the Concept class.

    A concept is a node with (optional) attributes and (optional) methods.
        Attributes and input, output of the methods can be other concepts,
        and are modeled as nodes.
    An input is a Graph with nodes.
    It is inherited from MultiDiGraph for graph manipulation and torch.nn.Module for
        gradient-based learning.
    """
    def __init__(self, G=None, **kwargs):
        """Components of a Graph instance:

        (1) A root node (name is given concept name), harboring a placeholder of the (instantiated)
            concept instance.
        (2) Its attributes, and arrows from the root node to the attributes. The attribute can be other Concept.
        (3) A binary-output function indicating if a given input is an instance of the concept. The function
            is a Graph() instance.
        """
        super(Concept, self).__init__(G=G, **kwargs)
        if "name" in kwargs:
            self.add_concept_def(definition=kwargs)
            if "inherit_from" in kwargs:
                self.inherit_from = kwargs["inherit_from"]
            if "inherit_to" in kwargs:
                self.inherit_to = kwargs["inherit_to"]


    @property
    def shape(self):
        value = self.get_node_value()
        if value is not None:
            return tuple(value.shape)
        else:
            return None


    def add_attr_recur(self, definition, base_node=None, neglect_obj=False):
        """Recursively construct the concept graph using definition (as a dictionary) or other defined concepts."""
        # Add attribute nodes:
        if not isinstance(definition, Concept):
            # Base definition:
            if "attr" in definition:
                base_node = self.name if base_node is None else base_node
                for key, value in definition["attr"].items():
                    if isinstance(value, tuple):
                        value, get_value = value
                        is_get_value = True
                    else:
                        is_get_value = False
                    attr_mode = str(value.mode).split("-")[0]
                    attr_name = "{}:{}".format(key, attr_mode)
                    self.add_node(attr_name, value=value, type="obj" if attr_mode==DEFAULT_OBJ_TYPE or ("is_all_obj" in definition and definition["is_all_obj"]) else "attr", fun=get_value if is_get_value else None)
                    if self.name is not None:
                        self.add_edge(base_node, attr_name, type="intra-attr")
                        # Each edges has a corresponding backward edge (with prefix "b-"), to facilitate traversing:
                        self.add_edge(attr_name, base_node, type="b-intra-attr")
                        self.add_attr_recur(CONCEPTS[attr_mode], base_node=attr_name)
        else:
            # Recursively adding attributes of attributes:
            base_node_key = base_node.split(":")[0]
            attr_names = definition.attributes
            for attr_name in attr_names:
                attr_node_name = "{}^{}".format(base_node_key, attr_name)
                attr_mode = attr_name.split(":")[-1]
                get_value = definition.nodes(data=True)[attr_name]["fun"] if "fun" in definition.nodes(data=True)[attr_name] else None
                self.add_node(attr_node_name, value=Placeholder(attr_mode), type="obj" if attr_mode==DEFAULT_OBJ_TYPE or ("is_all_obj" in definition and definition["is_all_obj"]) else "attr", fun=get_value)
                attr_value = definition.get_node_value(attr_name)
                if attr_value is not None:
                    self.set_node_value(attr_value, attr_node_name)
                self.add_edge(base_node, attr_node_name, type="intra-attr")
                self.add_edge(attr_node_name, base_node, type="b-intra-attr")
                if (not neglect_obj) or (definition.get_node_type(attr_name) != "obj"):
                    self.add_attr_recur(CONCEPTS[attr_mode], base_node=attr_node_name)


    def add_concept_def(self, definition):
        """Initialize the concept node from the concept definition dictionary."""
        name = definition["name"]
        assert "value" in definition
        if name is not None:
            self.add_node(name, value=definition["value"], type="concept")
            # Add representation:
            if "repr" in definition:
                self.nodes[name]["repr"] = nn.Parameter(definition["repr"])

        # Add attribute nodes:
        self.add_attr_recur(definition)

        # Add relations:
        if "re" in definition:
            for key, relation_name in definition["re"].items():
                if len(key) == 2:
                    source = self.get_node_name(key[0])
                    target = self.get_node_name(key[1])
                    self.add_edge(source, target, type="intra-relation", name=relation_name)


    def add_obj(
        self,
        obj,
        obj_name=None,
        change_root=True,
        add_full_concept=True,
        loc=None,
    ):
        """Add object attributes relative to the root node."""
        assert isinstance(obj, Concept)

        if obj_name is None:
            obj_name = get_next_available_key(self.nodes, "obj", suffix=":Image", is_underscore=True)
        self.add_node(obj_name, value=Placeholder(obj.name), type="obj")
        self.set_node_value(obj.get_node_value(), obj_name)
        if loc is None:
            if self.name is not None:
                self.add_edge(self.name, obj_name, type="intra-attr")
                self.add_edge(obj_name, self.name, type="b-intra-attr")
        else:
            loc = self.get_node_name(loc)
            self.add_edge(loc, obj_name, type="intra-attr")
            self.add_edge(obj_name, loc, type="b-intra-attr")
        if add_full_concept:
            self.add_attr_recur(obj, base_node=obj_name, neglect_obj=True)
        if change_root:
            root_tensor = self.get_node_value()
            self.set_node_value(set_patch(root_tensor, obj.get_node_value(), obj.get_node_value("pos")))
        return obj_name


    def add_attr(self, base_name, attr_mode):
        """Add an attribute with attr_mode on the node base_name."""
        base_name = self.get_node_name(base_name)
        base_op_name = self.operator_name(base_name)
        child_nodes = self.child_nodes(base_name)
        attr_name = get_next_available_key(child_nodes, "{}^{}".format(base_op_name, attr_mode.lower()), suffix=":{}".format(attr_mode), is_underscore=False, start_from_null=True)
        self.add_node(attr_name, value=Placeholder(attr_mode), type="obj" if attr_mode==DEFAULT_OBJ_TYPE else "attr")
        self.add_edge(base_name, attr_name, type="intra-attr")
        self.add_edge(attr_name, base_name, type="b-intra-attr")
        return self, attr_name


    def remove_attr_with_value(self, attr, change_root=True):
        """Remove an attribute and all its descendant attributes."""
        attr_name_find = None
        for attr_name in self.attributes:
            attr_value = self.get_node_value(attr_name)
            attr_value_given = attr.get_node_value()
            if tuple(attr_value.shape) == tuple(attr_value_given.shape) and (attr_value == attr_value_given).all():
                attr_name_find = attr_name
                break
        if attr_name_find is None:
            return self
        else:
            return self.remove_attr(attr_name_find, change_root=change_root)


    def remove_attr(self, attr_name, change_root=True):
        """Remove an attribute and all its descendant attributes."""
        descendants = self.get_descendants(attr_name, includes_self=True)
        if change_root:
            root_tensor = self.get_node_value()
            for node_name in descendants:
                if self.get_node_type(node_name) == "obj":
                    set_patch(root_tensor,
                              self.get_node_value(node_name),
                              self.get_node_value(self.operator_name(node_name) + "^pos"),
                              0)
        self.remove_nodes_from(descendants)
        return self


    def rename_nodes(self, mapping):
        """Rename nodes."""
        G = self.__class__(nx.relabel_nodes(self, mapping))
        if hasattr(self, "pivot_node_names"):
            G.pivot_node_names = self.pivot_node_names
        if hasattr(self, "refer_node_names"):
            G.refer_node_names = self.refer_node_names
        self.__dict__.update(G.__dict__)
        return self
        


    def combine_objs(self, obj_names):
        """Add an composite object node as parent for the obj_names.
        The obj_names must have the same parent.
        """
        obj_names = [self.get_node_name(name) for name in obj_names]
        parent_node = self.parent_nodes(obj_names[0])[0]
        # Check that the parent node is the same for all nodes in obj_names:
        for obj_name in obj_names[1:]:
            parent_node_ele = self.parent_nodes(obj_name)
            assert len(parent_node_ele) == 1 and parent_node_ele[0] == parent_node
        parent_node_op_name = self.operator_name(parent_node) + "^" if parent_node != self.name else ""
        comp_obj_name = get_next_available_key(self.obj_names, "{}obj".format(parent_node_op_name), suffix=":Image", is_underscore=True)
        comp_obj_op_name = self.operator_name(comp_obj_name)

        # Get composite obj:
        obj_dict = {obj_name: self.get_attr(obj_name) for obj_name in obj_names}
        comp_obj, pos_bounding = get_comp_obj(obj_dict, CONCEPTS)
        self.add_node(comp_obj_name, value=Placeholder(DEFAULT_OBJ_TYPE), type="obj")

        # Add attribute for the composite obj:
        self.add_node("{}^pos:Pos".format(comp_obj_op_name), value=Placeholder("Pos"), type="attr")
        self.add_node("{}^color:Color".format(comp_obj_op_name), value=Placeholder("Color"), type="attr", fun=self.get_node_fun("color"))
        self.set_node_value(comp_obj.get_node_value(), comp_obj_name)
        self.set_node_value(pos_bounding, "{}^pos:Pos".format(comp_obj_op_name))
        self.add_edge(comp_obj_name, "{}^pos:Pos".format(comp_obj_op_name), type="intra_attr")
        self.add_edge(comp_obj_name, "{}^color:Color".format(comp_obj_op_name), type="intra_attr")
        self.add_edge("{}^pos:Pos".format(comp_obj_op_name), comp_obj_name, type="b-intra_attr")
        self.add_edge("{}^color:Color".format(comp_obj_op_name), comp_obj_name, type="b-intra_attr")

        # Insert comp_obj as parent of obj_names:
        self.add_edge(parent_node, comp_obj_name, type="intra-attr")
        self.add_edge(comp_obj_name, parent_node, type="b-intra-attr")
        for obj_name in obj_names:
            self.remove_edge(obj_name, parent_node)
            self.remove_edge(parent_node, obj_name)
        for obj_name in obj_names:
            self.add_edge(comp_obj_name, obj_name, type="intra-attr")
            self.add_edge(obj_name, comp_obj_name, type="b-intra-attr")

        # Rename:
        rename_mapping = {}
        attrs_to_rename = self.get_descendants(obj_names, includes_self=True)
        for attr_name in attrs_to_rename:
            attr_branch_name = attr_name if parent_node == self.name else attr_name.split(parent_node_op_name)[1]
            rename_mapping[attr_name] = "{}^{}".format(comp_obj_op_name, attr_branch_name)
        self.rename_nodes(rename_mapping)
        return comp_obj_name


    def flatten_obj(self, obj_name):
        """Remove the obj_name and connect its children directly to its parents."""
        obj_name = self.get_node_name(obj_name)
        child_nodes = self.child_nodes(obj_name)
        parent_nodes = self.parent_nodes(obj_name)
        assert len(parent_nodes) == 1
        parent_node = parent_nodes[0]
        rename_mapping = {}
        obj_op_name = self.operator_name(obj_name)
        for child_node in child_nodes:
            if self.get_node_type(child_node) == "obj":
                descendants = self.get_descendants(child_node, includes_self=True)
                self.remove_edge(obj_name, child_node)
                self.remove_edge(child_node, obj_name)
                self.add_edge(parent_node, child_node, type="intra-attr")
                self.add_edge(child_node, parent_node, type="b-intra-attr")
                for name in descendants:
                    rename_mapping[name] = name.split(obj_op_name + "^")[1]
            else:
                self.remove_node(child_node)
        self.remove_node(obj_name)
        if hasattr(self, "pivot_node_names") and obj_name in self.pivot_node_names:
            self.pivot_node_names.remove(obj_name)
        if hasattr(self, "refer_node_names") and obj_name in self.refer_node_names:
            self.refer_node_names.remove(obj_name)
        self.rename_nodes(rename_mapping)
        return self


    def remove_relation_manual(self, obj1_name, obj2_name, relation_name):
        """Manually remove a specific relation."""
        obj1_name = self.get_node_name(obj1_name)
        obj2_name = self.get_node_name(obj2_name)
        has_effect = False
        if obj2_name in self[obj1_name]:
            for id, edge_info in self[obj1_name][obj2_name].copy().items():
                if edge_info["type"] == 'intra-relation' and edge_info["name"] == relation_name:
                    self.remove_edge(obj1_name, obj2_name, id)
                    has_effect = True
        return has_effect


    def add_relation_manual(self, relation_name, *opsc, **kwargs):
        """Manually add a specific relation. If it is a Concept_Pattern and its self.is_ebm is True, will
            also add a relation-EBM.

        Args:
            relation_name: type of the relation
            opsc: list of nodes that the relation will connect to in order.
        """
        if len(opsc) == 0:
            # Will have two new nodes:
            obj1_name = get_next_available_key([node_name.split(":")[0] for node_name in self.nodes], "obj", is_underscore=True)
            obj1_name = "{}:{}".format(obj1_name, DEFAULT_OBJ_TYPE)
            self.add_node(obj1_name, value=Placeholder(DEFAULT_OBJ_TYPE), type="obj")
            obj2_name = get_next_available_key([node_name.split(":")[0] for node_name in self.nodes], "obj", is_underscore=True)
            obj2_name = "{}:{}".format(obj2_name, DEFAULT_OBJ_TYPE)
            self.add_node(obj2_name, value=Placeholder(DEFAULT_OBJ_TYPE), type="obj")
        elif len(opsc) == 1:
            # Only provide the first object to connect to the first fun-in of the relation:
            obj1_name = self.get_node_name(opsc[0])
            obj2_name = get_next_available_key([node_name.split(":")[0] for node_name in self.nodes], "obj", is_underscore=True)
            obj2_name = "{}:{}".format(obj2_name, DEFAULT_OBJ_TYPE)
            self.add_node(obj2_name, value=Placeholder(DEFAULT_OBJ_TYPE), type="obj")
        elif len(opsc) == 2:
            if opsc[0] is not None:
                obj1_name = self.get_node_name(opsc[0])
            else:
                # the first opsc[0] is None, meaning that will initialize a new node with default concept:
                obj1_name = get_next_available_key([node_name.split(":")[0] for node_name in self.nodes], "obj", is_underscore=True)
                obj1_name = "{}:{}".format(obj1_name, DEFAULT_OBJ_TYPE)
                self.add_node(obj1_name, value=Placeholder(DEFAULT_OBJ_TYPE), type="obj")
            obj2_name = self.get_node_name(opsc[1])
        else:
            raise Exception("The length of ops can only be 0, 1 or 2!")
        if hasattr(self, "is_ebm") and self.is_ebm:
            if relation_name not in self.ebm_dict:
                self.init_ebm(
                    method="random",
                    mode=relation_name,
                    ebm_mode="operator",
                    ebm_model_type="CEBM",
                    **kwargs
                )
            placeholder = Placeholder(relation_name).set_ebm_key(relation_name)
            self.add_edge(obj1_name, obj2_name, type="intra-relation", name=relation_name, value=placeholder)
        else:
            self.add_edge(obj1_name, obj2_name, type="intra-relation", name=relation_name)
        # Important: reset the forward results cache
        if hasattr(self, "cache_forward") and self.cache_forward:
            self.forward_cache = {}
        return self


    def add_relation(self, obj1_name, obj2_name, OPERATORS, allowed_types=["Bool"]):
        """Find all the relations between obj1 and obj2 in both directions."""
        obj1_name = self.get_node_name(obj1_name)
        obj2_name = self.get_node_name(obj2_name)

        for key, op in OPERATORS.items():
            output_mode = op.get_to_outnode(op.name).split(":")[-1]
            if "Bool" in allowed_types and output_mode == "Bool":
                if "obj1" not in locals():
                    obj1 = self.get_attr(obj1_name)
                if "obj2" not in locals():
                    obj2 = self.get_attr(obj2_name)
                relations_exist = self.get_relation(obj1_name, obj2_name)

                if ((obj1_name, obj2_name) not in relations_exist or key not in relations_exist[(obj1_name, obj2_name)]) and check_input_valid(op, obj1_name, obj2_name):
                    is_valid = op(obj1, obj2)
                    if is_valid:
                        self.add_edge(obj1_name, obj2_name, type="intra-relation", name=key)
                if ((obj2_name, obj1_name) not in relations_exist or key not in relations_exist[(obj2_name, obj1_name)]) and check_input_valid(op, obj1_name, obj2_name):
                    is_valid = op(obj2, obj1)
                    if is_valid:
                        self.add_edge(obj2_name, obj1_name, type="intra-relation", name=key)
            elif "Op" in allowed_types and output_mode != "Bool" and len(op.dangling_nodes) == 0 and len(op.input_placeholder_nodes) == 1:
                relations_exist = self.get_relation(obj1_name, obj2_name)
                if "obj1" not in locals():
                    obj1 = self.get_attr(obj1_name)
                if "obj2" not in locals():
                    obj2 = self.get_attr(obj2_name)
                if ((obj1_name, obj2_name) not in relations_exist or key not in relations_exist[(obj1_name, obj2_name)]) and check_input_valid(op, obj1_name):
                    obj1_trans = op(obj1)
                    is_valid = obj1_trans == obj2
                    if is_valid:
                        self.add_edge(obj1_name, obj2_name, type="intra-relation", name=key)
                if ((obj2_name, obj1_name) not in relations_exist or key not in relations_exist[(obj2_name, obj1_name)]) and check_input_valid(op, obj2_name):
                    obj2_trans = op(obj2)
                    is_valid = obj2_trans == obj1
                    if is_valid:
                        self.add_edge(obj2_name, obj1_name, type="intra-relation", name=key)
        return self


    def add_relations(self, OPERATORS, allowed_types=["Bool"]):
        """Get valid relations between all pairs of objects."""
        node_lst = self.attributes
        is_obj = False if len(self.obj_names) == 0 else True
        if is_obj:
            node_lst = self.obj_names
        for i, obj1_name in enumerate(node_lst):
            for j, obj2_name in enumerate(node_lst):
                if i < j:
                    self.add_relation(obj1_name, obj2_name, OPERATORS, 
                                      allowed_types=allowed_types)
        return self


    def get_relation(self, obj1_name, obj2_name, bidirectional=True):
        """Get relation between two objects."""
        obj1_name = self.get_node_name(obj1_name)
        obj2_name = self.get_node_name(obj2_name)
        relations = {}
        if obj2_name in self[obj1_name]:
            for key, item in self[obj1_name][obj2_name].items():
                if item["type"] == "intra-relation":
                    record_data(relations, [item["name"]], [(obj1_name, obj2_name)])
        if not bidirectional:
            if (obj1_name, obj2_name) in relations:
                return relations[(obj1_name, obj2_name)]
            else:
                return []
        else:
            if obj1_name in self[obj2_name]:
                for key, item in self[obj2_name][obj1_name].items():
                    if item["type"] == "intra-relation":
                        record_data(relations, [item["name"]], [(obj2_name, obj1_name)])
            return relations


    def get_relations(self):
        """Get relation between each pair of objects."""
        node_lst = self.attributes
        is_obj = False if len(self.obj_names) == 0 else True
        if is_obj:
            node_lst = self.obj_names
        relations = {}
        for i, obj1_name in enumerate(node_lst):
            for j, obj2_name in enumerate(node_lst):
                if i < j:
                    relation = self.get_relation(obj1_name, obj2_name)
                    if len(relation) > 0:
                        relations.update(relation)
        return relations


    def get_neighbors(self, op_name):
        """Obtain the neighbor of an op_name.

        Args:
            op_name: if op_name is an op-sc, then its neighbors are op-so
                     if op_name is an op-so, then its neighbors are op-sc.
        """
        if ")" not in op_name:
            # op_name refers to a concept node, e.g. obj_1:Image:
            neighbors = []
            node1_name = self.get_node_name(op_name)
            for node2_name, edge_info in self[node1_name].items():
                for key, item in edge_info.items():
                    assert item["type"] == "intra-relation"
                    neighbors.append(f"({node1_name},{node2_name}):{item['name']}")
        else:
            # op_name refers to a relation edge, e.g. '(obj_1:c1,obj_2:c1):r1':
            neighbors = op_name.split(")")[0][1:].split(",")
        return neighbors


    def parse_comp_obj(self, comp_obj):
        """Given an comp_obj, find all the obj_names in self.objs that together forms the comp_obj"""
        comp_obj_value = comp_obj.get_node_value()
        comp_obj_pos = comp_obj.get_node_value("pos")
        unexplained_comp = comp_obj_value > 0
        obj_names_included = []
        for obj_name in self.obj_names:
            obj_value = self.get_node_value(obj_name)
            obj_pos = self.get_node_value(self.operator_name(obj_name) + "^pos")
            if obj_pos[0] >= comp_obj_pos[0] and obj_pos[1] >= comp_obj_pos[1] and obj_pos[0] + obj_pos[2] <= comp_obj_pos[0] + comp_obj_pos[2] and obj_pos[1] + obj_pos[3] <= comp_obj_pos[1] + comp_obj_pos[3]:
                # Object position is inside, then compare value:
                relpos = (int(obj_pos[0] - comp_obj_pos[0]), int(obj_pos[1] - comp_obj_pos[1]), int(obj_pos[2]), int(obj_pos[3]))
                patch = get_patch(comp_obj_value, relpos)  # The patch corresponding to the position of the obj
                mask_obj_g0 = obj_value > 0
                is_subset = (obj_value[mask_obj_g0] == patch[mask_obj_g0]).all()
                if is_subset:
                    obj_names_included.append(obj_name)
                    unexplained_comp[relpos[0]: relpos[0] + relpos[2], relpos[1]: relpos[1] + relpos[3]][mask_obj_g0] = 0
            else:
                intersect_pos = get_pos_intersection(obj_pos, comp_obj_pos)
                if intersect_pos is not None:
                    relpos = (int(intersect_pos[0] - obj_pos[0]),
                              int(intersect_pos[1] - obj_pos[1]),
                              int(intersect_pos[2]),
                              int(intersect_pos[3]))
                    comp_relpos = (int(intersect_pos[0] - comp_obj_pos[0]),
                                   int(intersect_pos[1] - comp_obj_pos[1]),
                                   int(intersect_pos[2]),
                                   int(intersect_pos[3]))
                    intersect_patch = get_patch(obj_value, relpos)
                    intersect_comp_patch = get_patch(comp_obj_value, comp_relpos)
                    is_subset = (intersect_comp_patch == intersect_patch).all()
                    if is_subset:
                        # Composite object and obj has overlap. Then split the object into intersect_obj and remainder object, 
                        # and add them as descendants to the obj:
                        remainder_obj_value = deepcopy(obj_value)
                        set_patch(remainder_obj_value, intersect_patch, relpos, 0)
                        remainder_obj_value, remainder_pos = shrink(remainder_obj_value)
                        remainder_obj = CONCEPTS[self.name].copy().set_node_value(remainder_obj_value)
                        remainder_obj.set_node_value([obj_pos[0] + remainder_pos[0],
                                                      obj_pos[1] + remainder_pos[1],
                                                      remainder_pos[2],
                                                      remainder_pos[3]],
                                                     "pos")
                        intersect_patch, intersect_shrink_pos = shrink(intersect_patch)
                        intersect_obj = CONCEPTS[self.name].copy().set_node_value(intersect_patch)
                        intersect_obj.set_node_value([intersect_pos[0] + intersect_shrink_pos[0],
                                                      intersect_pos[1] + intersect_shrink_pos[1],
                                                      intersect_shrink_pos[2],
                                                      intersect_shrink_pos[3]], 
                                                     "pos")
                        self.remove_attr(obj_name, change_root=False)
                        intersect_obj_name = self.add_obj(intersect_obj, change_root=False)
                        remainder_obj_name = self.add_obj(remainder_obj, change_root=False)
                        obj_names_included.append(intersect_obj_name)
                        unexplained_comp[comp_relpos[0]: comp_relpos[0] + comp_relpos[2], comp_relpos[1]: comp_relpos[1] + comp_relpos[3]][intersect_comp_patch > 0] = 0
            if unexplained_comp.sum() == 0:
                break
        return obj_names_included, unexplained_comp


    def get_concept_pattern_from_objs(self, comp_objs, is_self_contained=True):
        """Given a list of comp_objs, return a concept_pattern that together forms the comp_obj"""
        if not isinstance(comp_objs, list):
            comp_objs = [comp_objs]
        obj_names_all = []
        for comp_obj in comp_objs:
            obj_names, unexplained_comp = self.parse_comp_obj(comp_obj)
            assert unexplained_comp.sum() == 0
            obj_names_all += obj_names
        if is_self_contained:
            # The nodes in concept_pattern only come from the objects constituting the comp_objs:
            concept_pattern = self.get_concept_pattern(node_names=obj_names_all)
        else:
            # Includes all the objects in the graph, and the obj_names_all serves as refer_nodes:
            concept_pattern = self.get_concept_pattern(refer_node_names=obj_names_all)
        return concept_pattern


    def get_concept_pattern(
        self,
        node_names=None,
        pivot_node_names=None,
        refer_node_names=None,
    ):
        """Obtain Concept_Pattern instance."""
        G = deepcopy(self)
        node_names = [G.get_node_name(node_name) for node_name in node_names] if node_names is not None else G.nodes
        if pivot_node_names is None:
            if hasattr(self, "pivot_node_names"):
                pivot_node_names = self.pivot_node_names
        if refer_node_names is None:
            if hasattr(self, "refer_node_names"):
                refer_node_names = self.refer_node_names

        concept_pattern = Concept_Pattern(G.subgraph(node_names),
                                          pivot_node_names=pivot_node_names,
                                          refer_node_names=refer_node_names,
                                          parent_root_name=self.name,
                                         )
        return concept_pattern


    def get_matching_mapping(self, subconcept):
        """Given a subconcept (also an instance of Concept class), find the mapping of subgraph matching
        (that also obey the name of the relations), and return a list of mappings,
        each of which maps a node in subconcept to a node in self.
        """
        # convert to line graph to get edge-base isomorphism subgraphs
        DiGM = isomorphism.DiGraphMatcher(line_graph(DiGraph(self)), line_graph(DiGraph(subconcept)))
        p_relations = subconcept.get_relations()
        valid_mappings = []
        # No edges and only one node:
        if len(subconcept.edges) == 0:
            assert(len(subconcept.nodes) == 1 or len(subconcept.nodes) == 2)
            if len(subconcept.nodes) == 1:
                p_node = list(subconcept.nodes)[0]
                for node in self.nodes:
                    if node.split(":")[-1] == p_node.split(":")[-1] and node != self.name:
                        valid_mappings.append({p_node: node})
                return valid_mappings
            else:
                p_node1, p_node2 = list(subconcept.nodes)[0], list(subconcept.nodes)[1]
                for i in range(len(self.nodes)):
                    for j in range(i+1, len(self.nodes)):
                        node1 = list(self.nodes)[i]
                        node2 = list(self.nodes)[j]
                        if node1.split(":")[-1] == p_node1.split(":")[-1] and node1 != self.name                          and node2.split(":")[-1] == p_node2.split(":")[-1] and node2 != self.name:
                            valid_mappings.append({p_node1: node1, p_node2: node2})
                        elif node2.split(":")[-1] == p_node1.split(":")[-1] and node2 != self.name                          and node1.split(":")[-1] == p_node2.split(":")[-1] and node1 != self.name:
                            valid_mappings.append({p_node1: node2, p_node2: node1})        
                return valid_mappings
        
        # With edges:
        for edge_match in DiGM.subgraph_isomorphisms_iter():
            # convert edge maps to node maps
            mapping = {}
            is_valid = True
            for c_pair, p_pair in edge_match.items():
                if p_pair[0] in mapping:
                    if mapping[p_pair[0]] != c_pair[0] or c_pair[0].split(":")[-1] != p_pair[0].split(":")[-1]:
                        # Check if nodes are consistent:
                        is_valid = False
                        break
                elif c_pair[0] == self.name:
                    is_valid = False
                    break
                else:
                    mapping[p_pair[0]] = c_pair[0]
                if p_pair[1] in mapping:
                    if mapping[p_pair[1]] != c_pair[1] or c_pair[1].split(":")[-1] != p_pair[1].split(":")[-1]:
                        is_valid = False
                        break
                elif c_pair[1] == self.name:
                    is_valid = False
                    break
                else:
                    mapping[p_pair[1]] = c_pair[1]

                # Check if it is the same edge_type:
                p_edge_type = subconcept.get_edge_type(*p_pair)
                if p_edge_type != self.get_edge_type(*c_pair):
                    is_valid = False
                    break

                if p_edge_type == "intra-relation":
                    # Check if relation is a subset:
                    p_relation = p_relations[p_pair]
                    c_relation = self.get_relation(*c_pair, bidirectional=False)
                    if not set(p_relation).issubset(c_relation):
                        is_valid = False
                        break

            if is_valid:
                valid_mappings.append(mapping)
        return valid_mappings


    def query_node_names(self, subconcept, query_node_names, is_match_node=False):
        """Return the corresponding node names in self as the query_node_names in subconcept."""
        def get_is_match_node(mapping, refer_node_names):
            is_matched = True
            for p_node, c_node in mapping.items():
                if p_node not in refer_node_names:
                    if p_node.split(":")[-1] != c_node.split(":")[-1]:
                        is_matched = False
                        break
            return is_matched
        if not isinstance(query_node_names, list):
            query_node_names = [query_node_names]
        mappings = self.get_matching_mapping(subconcept)
        concept_node_names_list = []
        for mapping in mappings:
            # Make sure that each pivot node is matched:
            is_matched = True
            if is_match_node:
                is_matched = get_is_match_node(mapping, subconcept.refer_node_names)
                if not is_matched:
                    continue
            if hasattr(subconcept, "pivot_node_names") and subconcept.pivot_node_names is not None:
                for node_name in subconcept.pivot_node_names:
                    value_subconcept = subconcept.get_node_value(node_name)
                    node_name_concept = mapping[subconcept.get_node_name(node_name)]
                    value_concept = self.get_node_value(node_name_concept)
                    if value_subconcept.shape != value_concept.shape or not (value_subconcept == value_concept).all():
                        is_matched = False
                        break
            if not is_matched:
                continue
            # Obtain the concept_node_names:
            concept_node_names_list.append([mapping[subconcept.get_node_name(node_name)] for node_name in query_node_names])
        is_same_set = check_same_set(concept_node_names_list)
        if is_same_set is None:
            return is_same_set, []
        elif is_same_set is True:
            return is_same_set, concept_node_names_list[0]
        else:
            union_set = set(concept_node_names_list[0])
            for node_names in concept_node_names_list[1:]:
                union_set = union_set.union(set(node_names))
            return is_same_set, list(union_set)


    def get_pivot_node_names(self, concept_pattern, is_match_node=False):
        """Get pivot_node_names in self as specified by the pivot_node_names in concept_pattern"""
        if concept_pattern.pivot_node_names is not None:
            is_same_set, pivot_node_names = self.query_node_names(concept_pattern, concept_pattern.pivot_node_names, is_match_node=is_match_node)
            if not is_same_set:
                raise Exception("Different mappings return different sets of pivot nodes!")
        else:
            return []


    def get_pivot_nodes(self, concept_pattern, is_match_node=False):
        """Get pivot_nodes (in terms of concept instance) in self as specified by the pivot_node_names in concept_pattern"""
        pivot_node_names = self.get_pivot_node_names(concept_pattern, is_match_node=is_match_node)
        return {node_name: self.get_attr(node_name) for node_name in pivot_node_names}


    def get_refer_node_names(self, concept_pattern, is_match_node=False):
        """Get refer_node_names in self as specified by the refer_node_names in concept_pattern"""
        if concept_pattern.refer_node_names is None:
            refer_node_names = list(concept_pattern.nodes)
        else:
            refer_node_names = concept_pattern.refer_node_names
        is_same_set, refer_node_names = self.query_node_names(concept_pattern, refer_node_names, is_match_node=is_match_node)
        return refer_node_names


    def get_refer_nodes(self, concept_pattern, is_match_node=False):
        """Get refer_nodes (in terms of concept instance) in self as specified by the refer_node_names in concept_pattern"""
        refer_node_names = self.get_refer_node_names(concept_pattern, is_match_node=is_match_node)
        return {node_name: self.get_attr(node_name) for node_name in refer_node_names}


    def find_node(self, node_name):
        return any([node for node in self.nodes(data=True) if node == node_name])


    def get_refer_subconcept(self, concept_pattern):
        """Given a concept_pattern (also an instance of Concept class), find the mapping of subgraph matching
        (that also obey the name of the relations), and return a single concept
        that matches the concept_pattern (both nodes and relations)
        """
        p_relations = concept_pattern.get_relations()
        refer_nodes = concept_pattern.refer_node_names
        if refer_nodes is None:
            refer_nodes = list(concept_pattern.nodes)
        # Get all valid node mappings as a list of subgraphs
        node_mappings = self.get_matching_mapping(concept_pattern)
        subconcept = Concept(name=self.name, **self.root_node)
        for subgraph_map in node_mappings:
            # Pairs of pattern node, concept node
            curr_mapping = {}
            for p_node, c_node in subgraph_map.items():
                # Only keep the concept nodes that are also refer nodes
                if p_node in refer_nodes:
                    curr_mapping[p_node] = c_node
            # Go through all pairs of concepts in this subgraph and add edges
            for ind1, p_node1 in enumerate(curr_mapping.keys()):
                for ind2, p_node2 in enumerate(curr_mapping.keys()):
                    if ind1 < ind2:
                        c_node1 = curr_mapping[p_node1]
                        c_node2 = curr_mapping[p_node2]
                        # Add c_node1 and c_node2 if not already in subconcept
                        if c_node1 not in subconcept.nodes:
                            subconcept.add_node(c_node1, value = self.get_node_content(c_node1),
                                               type = self.get_node_type(c_node1),
                                               repr = self.get_node_repr(c_node1))
                            subconcept.add_edge(self.name, c_node1, type="intra-attr")
                            subconcept.add_edge(c_node1, self.name, type="b-intra-attr")
                        if c_node2 not in subconcept.nodes:
                            subconcept.add_node(c_node2, value = self.get_node_content(c_node2),
                                               type = self.get_node_type(c_node2),
                                               repr = self.get_node_repr(c_node2))
                            subconcept.add_edge(self.name, c_node2, type="intra-attr")
                            subconcept.add_edge(c_node2, self.name, type="b-intra-attr")
                        p_relation = p_relations[(p_node1, p_node2)]
                        for relation in p_relation:
                            subconcept.add_relation_manual(relation, c_node1, c_node2)
        print(subconcept.get_relations())
        return subconcept


    @property
    def root_node(self):
        """Get the content of the root node."""
        return self.nodes(data=True)[self.name]


    @property
    def obj_names(self):
        """Get object names."""
        nodes_sorted = self.topological_sort
        return deepcopy([node for node in nodes_sorted if self.nodes[node]["type"] == "obj" and node != self.name])


    @property
    def obj_names_aug(self):
        """Get object names, and if there is only root object, returns the root object."""
        obj_names = self.obj_names
        if len(obj_names) == 0:
            obj_names = ["$root"]
        return obj_names


    @property
    def objs(self):
        """Get a dictionary of {obj_name: obj}."""
        objs = OrderedDict()
        for obj_name in self.obj_names:
            objs[obj_name] = self.get_attr(obj_name)
        return objs


    def draw_objs(self):
        """Draw objects."""
        for obj_name, obj in self.objs.items():
            print("{}:".format(obj_name))
            obj.draw()


    @property
    def attributes(self):
        """Get all the attribute names of the concept."""
        if self.name is not None:
            return deepcopy(self.child_nodes(self.name))
        else:
            return list(self.nodes)


    def get_reprs(self, allowed_attr="all"):
        """Return the reprentations for the attributes in N x REPR_DIM, where N is the number of
        nodes in the graph, and each row corresponds to the global representation of the concept.
        """
        x = []
        for node in self.get_graph(allowed_attr).topological_sort:
            mode = node.split(":")[-1]
            x.append(CONCEPTS[mode].get_node_repr())
        x = torch.stack(x, 0)
        return x


    def get_edge_index_attr_de(self, OPERATORS, allowed_attr="all", repr_format="onehot"):
        """Return edge_index for descendant relations in COO format, where the nodes' index
            is according to self.topological_sort."""
        edge_index = []
        node_sorted = self.get_graph(allowed_attr).topological_sort
        for i, node in enumerate(node_sorted):
            for child_node in self.child_nodes(node):
                if child_node in node_sorted:
                    j = node_sorted.index(child_node)
                    edge_index.append([i, j])
        if len(edge_index) > 0:
            edge_index = to_Variable(edge_index).long().T.to(self.device)
        else:
            edge_index = torch.zeros(2, 0).long().to(self.device)
        if repr_format == "onehot":
            edge_attr = torch.zeros(edge_index.shape[-1], len(OPERATORS) + 4).to(self.device)
        elif repr_format == "embedding":
            edge_attr = torch.zeros(edge_index.shape[-1], REPR_DIM).to(self.device)
        else:
            raise Exception("repr_format {} is not valid!".format(repr_format))
        return edge_index, edge_attr


    def get_edge_index_attr_re(self, OPERATORS, allowed_attr="all", repr_format="onehot"):
        """Return edge_index for intra-attribute relations in COO format, where the nodes' index 
            is according to self.topological_sort."""
        edge_index = []
        edge_attr = []
        node_sorted = self.get_graph(allowed_attr).topological_sort
        relations_dict = self.get_relations()
        for (source, target), relations in relations_dict.items():
            source_id = node_sorted.index(source)
            target_id = node_sorted.index(target)
            for relation in relations:
                edge_index.append([source_id, target_id])
                # Relation 0 is preserved for descendant relations
                if repr_format == "onehot":
                    edge_attr_tensor = torch.zeros(len(OPERATORS) + 4)
                    op_id = list(OPERATORS.keys()).index(relation)
                    edge_attr_tensor[op_id] = 1
                elif repr_format == "embedding":
                    edge_attr_tensor = OPERATORS[relation].get_node_repr()
                else:
                    raise Exception("repr_format {} is not valid!".format(repr_format))
                edge_attr.append(edge_attr_tensor)
        if len(edge_index) > 0:
            edge_index = to_Variable(edge_index).long().T.to(self.device)
            edge_attr = torch.stack(edge_attr).to(self.device)
        else:
            edge_index = torch.zeros(2, 0).long().to(self.device)
            edge_attr = torch.zeros(0, len(OPERATORS) + 4 if repr_format == "onehot" else REPR_DIM).float().to(self.device)
        return edge_index, edge_attr


    def get_PyG_data(self, OPERATORS=None, allowed_attr="all", repr_format="onehot"):
        """Get the graph data in PyG format.
        Args:
            OPERATORS: if not None, the edge_index and edge_attr will include the relational edge data.
            allowed_attr: choose from "all" (allowing all attribute nodes) and "obj" (only allowing object nodes).

        Attributes: x: global reprenstation for each operator, where each row is for one operator 
                                sorted by self.operators_core.
                    edge_index: edge_index for operators in COO format, where the operators' index 
                                is according to self.operators_core
                    edge_attr: edge_attr with shape [N, 1]. 0 means descendant relation. Other integers indicate
                                the index of relation in OPERATORS.
        """
        from torch_geometric.data import Data
        edge_index_de, edge_attr_de = self.get_edge_index_attr_de(OPERATORS, allowed_attr=allowed_attr, repr_format=repr_format)
        edge_index_re, edge_attr_re = self.get_edge_index_attr_re(OPERATORS, allowed_attr=allowed_attr, repr_format=repr_format)
        edge_index = torch.cat([edge_index_de, edge_index_re], -1)
        edge_attr = torch.cat([edge_attr_de, edge_attr_re])
        data = Data(x=self.get_reprs(allowed_attr), edge_index=edge_index, edge_attr=edge_attr)
        return data


    def get_root_value(self):
        """Get the value of the root node."""
        return self.get_node_value(self.name)


    def get_subgraph(self, nodes, includes_root=False, includes_descendants=True):
        """Get a subgraph using nodes and their descendants."""
        if includes_descendants:
            nodes_to_preserve = []
            for node in nodes:
                nodes_to_preserve += self.get_descendants(node, includes_self=True)
        else:
            nodes_to_preserve = nodes
        if includes_root:
            nodes_to_preserve.append(self.name)
        nodes_to_preserve = list(set(nodes_to_preserve))
        G = Concept(self.subgraph(nodes_to_preserve))
        G.name = self.name if includes_root else None
        return G


    def get_attr(self, attr_name):
        """Get the full attribute (preserving concept form and all its descendants)."""
        if attr_name == "$root":
            return self
        G = self.get_subgraph(self.get_descendants(attr_name, includes_self=True), includes_root=False)

        # Rename root node:
        attr_prefix = attr_name.split(":")[0]
        length = len(attr_prefix)
        mapping = {}
        for node_name in G.nodes:
            mapping[node_name] = node_name[length + 1:]

        G = nx.relabel_nodes(G, mapping)
        G.name = attr_name.split(":")[-1]
        return G


    def get_attr_value(self, attr_name):
        """Get the value of an attribute node."""
        return self.get_node_value(attr_name)


    def set_node_value(self, value, node_name=None):
        """Set up the value held in the Placeholder of a node, if the content is a Placeholder.
        Then compute all the attributes belonging to its descendants
        """
        assert isinstance(self.get_node_content(node_name), Placeholder)
        node_name = self.get_node_name(node_name)
        if not isinstance(value, torch.Tensor):
            self.nodes(data=True)[node_name]["value"].value = to_Variable(value, is_cuda=self.is_cuda)
        else:
            self.nodes(data=True)[node_name]["value"].value = value
        self.compute_attr_value(node_name=node_name)
        return self


    def compute_attr_value(self, node_name=None):
        """Calculate the attribute values of the descendants of node_name"""
        node_name = self.get_node_name(node_name)
        node_value = self.get_node_value(node_name)
        if node_value is not None:
            for child_node in self.child_nodes(node_name):
                fun = self.get_node_fun(child_node)
                if fun is not None:
                    if self.get_node_value(child_node) is None:
                        self.set_node_value(fun(node_value), child_node)
        return self


    def draw(self, is_clean_graph=True, layout="spring", filename=None, **kwargs):
        """Visualize the current graph."""
        # Only plot the acyclic edges, and by default the forward graph:
        import logging
        logging.getLogger('matplotlib.font_manager').disabled = True
        if is_clean_graph:
            G = self.clean_graph(is_copy_module=False)
        else:
            G = self.forward_graph(is_copy_module=False) if not nx.is_directed_acyclic_graph(self) else self
        # Nodes:
        node_color = []
        node_sizes = []
        node_list = []
        attr_node_list = []

        for node, info in G.nodes(data=True):
            node_size = 1200
            if info["type"] == "attr" and self.__class__.__name__ != "Concept_Pattern":
                attr_node_list.append(node)
                continue
            node_list.append(node)
            if info["type"] == "input":
                node_color.append("#58509d")
            elif info["type"] == "concept":
                node_color.append("#C515E8")
                if self.get_node_value(node) is not None:
                    node_size = 1800
            elif info["type"] == "self":
                node_color.append("#1f78b4")
            elif info["type"] == "obj":
                if hasattr(self, "pivot_node_names") and self.pivot_node_names is not None and node in self.pivot_node_names:
                    node_color.append("#FA49E2")
                elif hasattr(self, "refer_node_names") and self.refer_node_names is not None and node in self.refer_node_names:
                    node_color.append("#FA9968")
                else:
                    node_color.append("#C31B37")
            elif info["type"] == "attr":
                node_color.append("#EB8F8F")
            elif info["type"].startswith("fun"):
                if info["type"] == "fun-out":
                    node_color.append("orange")
                else:
                    node_color.append("g")
            else:
                raise
            node_sizes.append(node_size)

        # Edges:
        edge_color = []
        edge_list = []
        for ni, no, data in G.edges(data=True):
            if no in attr_node_list:
                continue
            edge_list.append((ni, no))
            if "intra-relation" in data["type"]:
                edge_color.append("purple")
            elif "intra" in data["type"]:
                edge_color.append("k")
            elif "inter" in data["type"]:
                if data["type"].endswith("inter-input"):
                    edge_color.append("brown")
                elif data["type"].endswith("inter-criteria"):
                    edge_color.append("c")
                else:
                    raise
            elif data["type"].endswith("get-attr"):
                edge_color.append("#E815DA")
            else:
                raise
        # Set up layout:
        if layout == "planar":
            pos = nx.planar_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        elif layout == "spiral":
            pos = nx.spiral_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G)
        elif layout == "kk":
            pos = nx.kamada_kawai_layout(G)
        else:
            raise

        # Draw:
        G.remove_nodes_from(attr_node_list)
        nx.draw(G, with_labels=True, font_size=10, pos=pos, alpha=0.8 if isinstance(self, Concept_Pattern) else 1,
                nodelist=node_list, node_color=node_color, node_size=node_sizes,
                edgelist=edge_list, edge_color=edge_color,
               )

        # Draw edge labels:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={edge: ",".join(relation_list) for edge, relation_list in self.get_relations().items()},
            font_color='red',
        )

        # Recursively draw composite concepts:
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", **kwargs)
        else:
            plt.show()
            if isinstance(self, Concept) and self.name is not None:
                value = self.get_node_value(self.name)
                if value is not None:
                    if len(value.shape) == 2:
                        visualize_matrices([value])
                    elif len(value.shape) == 3:
                        value_T = deepcopy(value).permute(1,2,0)
                        plt.imshow(to_np_array(value_T).astype(int))
                        plt.show()

            # If certain node's content is a graph, recursively draw it:
            for node in self.nodes:
                node_content = self.get_node_content(node)
                if isinstance(node_content, BaseGraph):
                    print("\nDrawing the content of node '{}', which is {}:".format(node, node_content))
                    node_content.draw()
        return self


    def parse(self, input, Sobj, abs_pos=(0,0), score_mode="mean", **kwargs):
        """Parse an input image (or CONCEPTS[DEFAULT_OBJ_TYPE]), and see if it satisfies
        all the relations required by the current concept.
        
        Args:
            input: A Concept() instance with its attribute of absolute position,
                or a PyTorch Tensor for the value of the concept, in which case 
                the abs_pos must be given.
            Sobj: function to select objects from a scene.
            abs_pos: the absolute position of the input, in the case where input
                is a PyTorch tensor.
        """
        def assign_key_to_concept_inst(key_dict, concept_inst_dict):
            """
            Assign concept instances to corresponding keys.

            Example:
                key_dict:          {"Line": ["line1:Line", "line2:Line"]}
                concept_inst_dict: {"Line": [Line(), Line()]}

            Return:
                all_dictL          {"line1:Line": Line(), "line2:Line": Line()}
            """
            all_dict = OrderedDict()
            assert set(key_dict) == set(concept_inst_dict)
            for key, key_ele_list in key_dict.items():
                concept_inst_list = concept_inst_dict[key]
                if len(key_ele_list) > len(concept_inst_list):
                    return all_dict
                for key_ele, concept_inst in zip(key_ele_list, concept_inst_list):
                    all_dict[key_ele] = concept_inst
            return all_dict

        if isinstance(input, Concept):
            abs_pos = to_np_array(input.get_node_value("pos")[:2]).round().astype(int) if input.get_node_value("pos") is not None else (0, 0)
            input = input.get_root_value()
            # The input is a Tensor.
        score_dict = {}
        is_valid = True
        pos_all_dict = OrderedDict()
        if not hasattr(self, "re"):
            is_valid = False
        else:
            for key, relation in self.re.items():
                if isinstance(key, tuple):
                    # The relation concerns two or more attributes:
                    concept_inst_dict = OrderedDict()
                    key_dict = OrderedDict()
                    for key_ele in key:
                        key_ele = self.get_node_name(key_ele)
                        mode = key_ele.split(":")[-1]
                        if mode not in key_dict:
                            key_dict[mode] = [key_ele]
                        else:
                            key_dict[mode].append(key_ele)
                        if mode not in concept_inst_dict:
                            concept_instances = Sobj(input, CONCEPTS[mode], abs_pos=abs_pos)
                            # Get pos of all the concept_instances:
                            if not isinstance(concept_instances, dict):
                                concept_instances = OrderedDict([["Sobj-0", concept_instances]])
                            # Save to concept_inst_dict:
                            concept_inst_dict[mode] = list(concept_instances.values())
                    assigned_dict = assign_key_to_concept_inst(key_dict, concept_inst_dict)
                    pos_all_dict.update(OrderedDict([[key_part.split(":")[0], instance.get_node_value("pos")] for key_part, instance in assigned_dict.items()]))
                    if len(assigned_dict) == 0:
                        return False, score_dict, None
                    score = relation(*[assigned_dict[self.get_node_name(key_ele)] for key_ele in key])
                    if isinstance(score, Concept):
                        score = score.get_root_value()
                else:
                    # The relation concerns only one attribute or itself:
                    if key == "self":
                        score = relation(input)
                        pos_all_dict["self"] = (abs_pos[0], abs_pos[1], input.shape[0], input.shape[1])
                    else:
                        score = relation(kwargs[key])
                if isinstance(score, Concept):
                    score = to_np_array(score.get_root_value())
                score_dict[key] = score
                is_valid = is_valid and (score >= 1.)
        if is_valid:
            if len(pos_all_dict) == 0:
                pos_all = [[abs_pos[0], abs_pos[1], input.shape[0], input.shape[1]]]  # In case that it is a single image.
            else:
                pos_all = list(pos_all_dict.values())
            G = self.copy()
            pos_union = combine_pos(*pos_all)
            G.set_node_value(pos_union, "pos:Pos")
            G.set_node_value(input[pos_union[0] - abs_pos[0]: pos_union[0] + pos_union[2] - abs_pos[0], 
                                   pos_union[1] - abs_pos[1]: pos_union[1] + pos_union[3] - abs_pos[1]], G.name)
            if "assigned_dict" in locals():
                for key, item in assigned_dict.items():
                    G.set_node_value(item.get_root_value(), key)
            for key, pos in pos_all_dict.items():
                if key != "self":
                    G.set_node_value(pos, "{}^pos".format(key))
            for key, item in kwargs.items():
                G.set_node_value(item, key)
        else:
            G = None
        score = get_generalized_mean(list(score_dict.values()), cumu_mode=score_mode)
        return is_valid, (score, score_dict), G


    @property
    def DL(self):
        """Description length of the concept."""
        return len(self.attributes) + 1 + self.edge_index.shape[1]


    # Overloading mathematical operations:
    def __bool__(self):
        """True (or False)"""
        return self.get_node_content().__bool__()


    def __gt__(self, concept2):
        """Greater than (>). Both the self and concept2 must have the same root shape. Return a BoolTensor of the same shape."""
        tensor_self = self.get_node_value()
        tensor_other = concept2.get_node_value() if isinstance(concept2, Concept) else concept2
        tensor_gt = tensor_self > tensor_other
        G = self.copy()
        G.set_node_value(tensor_gt)
        for obj_name in self.obj_names:
            tensor_obj_other = concept2.get_node_value(obj_name) if isinstance(concept2, Concept) else concept2
            tensor_obj_gt = self.get_node_value(obj_name) > tensor_obj_other
            G.set_node_value(tensor_obj_gt, obj_name)
        return G


    def __lt__(self, concept2):
        """Less than (<). Both the self and concept2 must have the same root shape. Return a BoolTensor of the same shape."""
        tensor_self = self.get_node_value()
        tensor_other = concept2.get_node_value() if isinstance(concept2, Concept) else concept2
        tensor_lt = tensor_self < tensor_other
        G = self.copy()
        G.set_node_value(tensor_lt)
        for obj_name in self.obj_names:
            tensor_obj_other = concept2.get_node_value(obj_name) if isinstance(concept2, Concept) else concept2
            tensor_obj_lt = self.get_node_value(obj_name) < tensor_obj_other
            G.set_node_value(tensor_obj_lt, obj_name)
        return G


    def __eq__(self, concept2):
        """Equal to (==). Return a Bool(True/False) concept."""
        tensor_self = self.get_node_value()
        tensor_other = concept2.get_node_value() if isinstance(concept2, Concept) else concept2
        if tuple(tensor_self.shape) == tuple(tensor_other.shape) and (tensor_self == tensor_other).all():
            tensor_eq = torch.BoolTensor([True])[0]
        else:
            tensor_eq = torch.BoolTensor([False])[0]
        C = CONCEPTS["Bool"].copy().set_node_value(tensor_eq)
        return C


    # Printing:
    def __str__(self):
        repr_str = self.graph["name"] if "name" in self.graph else "Concept"
        # Composing content string:
        content_str = ""
        attr_node_list = self.obj_names
        if len(attr_node_list) > 0:
            for attr_node in attr_node_list:
                content_str += "{}, ".format(attr_node)
            content_str = content_str[:-2]
        if repr_str == "Bool":
            content_str = str(to_np_array(self.get_root_value()))
        return '{}({})'.format(repr_str, content_str)


    def __repr__(self):
        if IS_VIEW:
            if len(self.nodes) > 0:
                self.draw()
            if hasattr(self, "re"):
                for key, relation in self.re.items():
                    if isinstance(relation, Graph):
                        print("The relation on {} is {}, as follows:".format(key, relation))
                        relation.draw()
        return self.__str__()


    def get_string_repr(self, mode="obj"):
        """Get 1D string representation of the concept."""
        if mode == "obj":
            combined_dict = OrderedDict()
            combined_dict[self.name] = tensor_to_string(self.get_node_value())
            if self.name.split(":")[-1] == DEFAULT_OBJ_TYPE:
                combined_dict["pos"] = tensor_to_string(self.get_node_value("pos"))
            for obj_name in self.obj_names:
                combined_dict[obj_name] = tensor_to_string(self.get_node_value(obj_name))
                pos_obj = self.operator_name(obj_name) + "^pos"
                combined_dict[pos_obj] = tensor_to_string(self.get_node_value(pos_obj))
            string = ""
            for key, item in combined_dict.items():
                string += "#{}${}".format(key, item)
        else:
            raise Exception("mode {} is not valid!".format(mode))
        return string


# ### 1.4.2 Concept_Pattern

# In[ ]:


class Concept_Pattern(Concept):
    """Concept_Pattern (inherited from Concept) for refering to a subset of objects in a concept graph. In addition to Concept, it has 
    a pivot node(s) and pivot edges, and a subset of refer_nodes. The pivot node(edge) is for identifying
    and pivotting the Concept_Pattern within a larger concept graph, and refer nodes are the nodes being referred to.
    All edges are pivot edges.
    """
    def __init__(
        self,
        G=None,
        pivot_node_names=None,
        refer_node_names=None,
        parent_root_name=None,
        is_all_obj=False,
        is_ebm=False,
        is_default_ebm=False,
        is_selector_gnn=False,
        ebm_dict=None,
        gnn=None,
        CONCEPTS=None,
        OPERATORS=None,
        cache_forward=False,
        in_channels=10,
        # EBM specific:
        z_mode="None",
        z_first=2,
        z_dim=4,
        w_type="image+mask",
        mask_mode="concat",
        aggr_mode="max",
        pos_embed_mode="None",
        is_ebm_share_param=True,
        is_relation_z=False,
        img_dims=2,
        is_spec_norm=True,
        act_name="leakyrelu0.2",
        normalization_type="None",
        # Selector specific:
        channel_coef=None,
        empty_coef=None,
        obj_coef=None,
        mutual_exclusive_coef=None,
        pixel_entropy_coef=None,
        pixel_gm_coef=None,
        iou_batch_consistency_coef=None,
        iou_concept_repel_coef=None,
        iou_relation_repel_coef=None,
        iou_relation_overlap_coef=None,
        iou_attract_coef=None,
        SGLD_is_anneal=None,
        SGLD_is_penalize_lower=None,
        SGLD_mutual_exclusive_coef=None,
        SGLD_pixel_entropy_coef=None,
        SGLD_pixel_gm_coef=None,
        SGLD_iou_batch_consistency_coef=None,
        SGLD_iou_concept_repel_coef=None,
        SGLD_iou_relation_repel_coef=None,
        SGLD_iou_relation_overlap_coef=None,
        SGLD_iou_attract_coef=None,
        lambd_start=None,
        lambd=None,
        image_value_range=None,
        w_init_type=None,
        indiv_sample=None,
        step_size=None,
        step_size_img=None,
        step_size_z=None,
        step_size_zgnn=None,
        step_size_wtarget=None,
        **kwargs
    ):
        """
        Concept_Pattern as a partially instantiated Concept, for selecting object(s) within a Concept instance.

        Args:
            is_default_ebm: if True, the DEFAULT_OBJ_TYPE will also has its EBM.
        """
        super(Concept_Pattern, self).__init__(G=G, is_all_obj=is_all_obj, **kwargs)
        if parent_root_name is not None:
            rename_mapping = {}
            for node_name in self.nodes:
                if node_name.startswith("{}^".format(parent_root_name)):
                    rename_mapping[node_name] = node_name.split("{}^".format(parent_root_name))[1]
            G = nx.relabel_nodes(self, rename_mapping)
            self.__dict__.update(G.__dict__)
        self.name = None
        if G is not None and G.device is not None:
            self.device = G.device
        else:
            self.device = kwargs["device"] if "device" in kwargs else torch.device("cpu")
        for node_name in self.nodes:
            if "fun" in self.nodes(data=True)[node_name]:
                self.nodes(data=True)[node_name]['fun'] = None
        if pivot_node_names is not None:
            self.set_pivot_nodes(pivot_node_names)
        else:
            self.pivot_node_names = pivot_node_names
        if refer_node_names is not None:
            self.set_refer_nodes(refer_node_names)
        else:
            self.refer_node_names = refer_node_names
        self.in_channels = in_channels
        # EBM specific:
        self.z_mode = z_mode
        self.z_first = z_first
        self.z_dim = z_dim
        self.w_type = w_type
        self.mask_mode = mask_mode
        self.aggr_mode = aggr_mode
        self.pos_embed_mode = pos_embed_mode
        self.is_ebm_share_param = is_ebm_share_param
        self.is_relation_z = is_relation_z
        self.img_dims = img_dims
        self.is_spec_norm = is_spec_norm
        self.act_name = act_name
        self.normalization_type = normalization_type
        # selector specific:
        self.channel_coef = channel_coef
        self.empty_coef = empty_coef
        self.obj_coef = obj_coef
        self.mutual_exclusive_coef = mutual_exclusive_coef
        self.pixel_entropy_coef = pixel_entropy_coef
        self.pixel_gm_coef = pixel_gm_coef
        self.iou_batch_consistency_coef = iou_batch_consistency_coef
        self.iou_concept_repel_coef = iou_concept_repel_coef
        self.iou_relation_repel_coef = iou_relation_repel_coef
        self.iou_relation_overlap_coef = iou_relation_overlap_coef
        self.iou_attract_coef = iou_attract_coef
        self.SGLD_is_anneal = SGLD_is_anneal
        self.SGLD_is_penalize_lower = SGLD_is_penalize_lower
        self.SGLD_mutual_exclusive_coef = SGLD_mutual_exclusive_coef
        self.SGLD_pixel_entropy_coef = SGLD_pixel_entropy_coef
        self.SGLD_pixel_gm_coef = SGLD_pixel_gm_coef
        self.SGLD_iou_batch_consistency_coef = SGLD_iou_batch_consistency_coef
        self.SGLD_iou_concept_repel_coef = SGLD_iou_concept_repel_coef
        self.SGLD_iou_relation_repel_coef = SGLD_iou_relation_repel_coef
        self.SGLD_iou_relation_overlap_coef = SGLD_iou_relation_overlap_coef
        self.SGLD_iou_attract_coef = SGLD_iou_attract_coef
        self.lambd_start = lambd_start
        self.lambd = lambd
        self.image_value_range = image_value_range
        self.w_init_type = w_init_type
        self.indiv_sample = indiv_sample
        self.step_size = step_size
        self.step_size_img = step_size_img
        self.step_size_z = step_size_z
        self.step_size_zgnn = step_size_zgnn
        self.step_size_wtarget = step_size_wtarget

        # When cache_forward is True, store a dictionary of input image hash to
        # forward_NN results
        self.cache_forward = cache_forward
        if G is not None:
            self.forward_cache = G.forward_cache
        else:
            self.forward_cache = {} if self.cache_forward else None

        self.is_all_obj = is_all_obj
        self.is_ebm = is_ebm
        self.is_default_ebm = is_default_ebm
        self.is_selector_gnn = is_selector_gnn
        if self.is_ebm:
            if ebm_dict is not None:
                if self.is_ebm_share_param:
                    if len(ebm_dict) == 0:
                        self.ebm_dict = Shared_Param_Dict(is_relation_z=self.is_relation_z)
                    else:
                        self.ebm_dict = ebm_dict
                else:
                    self.ebm_dict = Combined_Dict(ebm_dict).set_is_relation_z(self.is_relation_z)
            self.CONCEPTS = CONCEPTS
            self.OPERATORS = OPERATORS
            self.init_ebms(
                method="random",
                ebm_model_type="CEBM",
                CONCEPTS=self.CONCEPTS,
                OPERATORS=self.OPERATORS,
                **kwargs
            )
            if self.is_selector_gnn:
                self.gnn = gnn
                self.zgnn_dim = gnn.zgnn_dim
                self.edge_attr_size = gnn.edge_attr_size


    def copy_with_grad(self, is_share_fun=False, is_copy_module=True, global_attrs=None):
        """Return the copy of current instance by detaching tensors which have grad
        and deepcopying all other object attributes.

        Args:
            is_share_fun: if True, the copy will share its torch.nn.Modules with its original.
            is_copy_module: if True, will copy torch.nn.Module. Otherwise not copy.
            global_attrs: a list of class attribute names that are global dictionaries.

        Returns:
            G: the copied class instance.
        """
        G_copy = self.__class__()
        copied_dict, global_dicts = copy_helper(self.__dict__, is_copy_module=is_copy_module, global_attrs=global_attrs)
        G_copy.__dict__.update(copied_dict)
        G = self.__class__(
            G=G_copy,
            pivot_node_names=deepcopy(self.pivot_node_names),
            refer_node_names=deepcopy(self.refer_node_names),
            is_all_obj=self.is_all_obj,
            is_ebm=self.is_ebm,
            is_default_ebm=self.is_default_ebm,
            is_selector_gnn=self.is_selector_gnn,
            gnn=self.gnn,
            ebm_dict=self.ebm_dict,
            CONCEPTS=self.CONCEPTS,
            OPERATORS=self.OPERATORS,
            cache_forward=self.cache_forward,
            in_channels=self.in_channels,
            z_mode=self.z_mode,
            z_first=self.z_first,
            z_dim=self.z_dim,
            w_type=self.w_type,
            mask_mode=self.mask_mode,
            aggr_mode=self.aggr_mode,
            pos_embed_mode=self.pos_embed_mode,
            is_ebm_share_param=self.is_ebm_share_param,
            is_relation_z=self.is_relation_z,
            img_dims=self.img_dims,
            is_spec_norm=self.is_spec_norm,
            act_name=self.act_name,
            normalization_type=self.normalization_type,
            channel_coef=self.channel_coef,
            empty_coef=self.empty_coef,
            obj_coef=self.obj_coef,
            mutual_exclusive_coef=self.mutual_exclusive_coef,
            pixel_entropy_coef=self.pixel_entropy_coef,
            pixel_gm_coef=self.pixel_gm_coef,
            iou_batch_consistency_coef=self.iou_batch_consistency_coef,
            iou_concept_repel_coef=self.iou_concept_repel_coef,
            iou_relation_repel_coef=self.iou_relation_repel_coef,
            iou_relation_overlap_coef=self.iou_relation_overlap_coef,
            iou_attract_coef=self.iou_attract_coef,
            SGLD_is_anneal=self.SGLD_is_anneal,
            SGLD_is_penalize_lower=self.SGLD_is_penalize_lower,
            SGLD_mutual_exclusive_coef=self.SGLD_mutual_exclusive_coef,
            SGLD_pixel_entropy_coef=self.SGLD_pixel_entropy_coef,
            SGLD_pixel_gm_coef=self.SGLD_pixel_gm_coef,
            SGLD_iou_batch_consistency_coef=self.SGLD_iou_batch_consistency_coef,
            SGLD_iou_concept_repel_coef=self.SGLD_iou_concept_repel_coef,
            SGLD_iou_relation_repel_coef=self.SGLD_iou_relation_repel_coef,
            SGLD_iou_relation_overlap_coef=self.SGLD_iou_relation_overlap_coef,
            SGLD_iou_attract_coef=self.SGLD_iou_attract_coef,
            lambd_start=self.lambd_start,
            lambd=self.lambd,
            image_value_range=self.image_value_range,
            w_init_type=self.w_init_type,
            indiv_sample=self.indiv_sample,
            step_size=self.step_size,
            step_size_img=self.step_size_img,
            step_size_z=self.step_size_z,
            step_size_zgnn=self.step_size_zgnn,
            step_size_wtarget=self.step_size_wtarget,
        )

        if global_attrs is not None:
            # Set the global dictionaries as attributes of the class:
            for key, Dict in global_dicts.items():
                setattr(G, key, Dict)
        return G


    def set_cache_forward(self, cache_forward):
        """Set the cache_forward attribute."""
        self.cache_forward = cache_forward
        if cache_forward is True:
            if self.forward_cache is None:
                self.forward_cache = {}
        elif cache_forward is False:
            self.forward_cache = None
        else:
            raise Exception("cache_forward must be True or False!")


    def init_ebms(
        self,
        method="random",
        ebm_model_type="CEBM",
        CONCEPTS=None,
        OPERATORS=None,
        ebm_dict=None,
        **kwargs
    ):
        """Initialize the EBMs for the Concept_Pattern.

        Args:
            mode: choose from "trained" (loading from best trained EBMs), 
                and "random" (initialize from random parameters or load from ebm_dict)
            ebm_model_type: model_type for the EBMs. Choose from "CEBM", "ConjEBM".
            ebm_dict: if not None, will contain some already-existing EBMs, whose key are the concepts/relation/operator type.
                Only effective if mode=="random".
            kwargs: init parameters for the EBM model if mode == "random".

        Returns:
            ebm_dict: A dictionary of EBMs, whose keys are the concept/relation/operator's type.
        """

        # Update the self.ebm_dict with the gievn ebm_dict. If not exist, create one:
        self.is_ebm = True
        if ebm_dict is None:
            if not hasattr(self, "ebm_dict"):
                self.ebm_dict = Shared_Param_Dict() if self.is_ebm_share_param else Combined_Dict()
        else:
            assert hasattr(self, "ebm_dict")
            self.ebm_dict.update(ebm_dict)

        # Setting up the EBMs for concept nodes:
        for node in self.nodes:
            placeholder = self.get_node_content(node)
            placeholder.ebm_key = self.init_ebm(
                method=method,
                mode=placeholder.mode,
                ebm_mode="concept",
                ebm_model_type=ebm_model_type,
                CONCEPTS=CONCEPTS if CONCEPTS is not None else self.CONCEPTS,
                **kwargs
            )

        # Setting up the EBMs for the relation nodes:
        for node_source, node_target, info in self.edges(data=True):
            assert info["type"] == "intra-relation"
            relation_name = info["name"]
            placeholder = Placeholder(mode=relation_name)
            info["value"] = placeholder
            placeholder.ebm_key = self.init_ebm(
                method=method,
                mode=placeholder.mode,
                ebm_mode="operator",
                ebm_model_type=ebm_model_type,
                OPERATORS=OPERATORS if OPERATORS is not None else self.OPERATORS,
                **kwargs
            )
        return self


    def init_ebm(self, method, mode, ebm_mode, ebm_model_type, CONCEPTS=None, OPERATORS=None, **kwargs):
        """Initialize the EBMs for the Concept_Pattern.

        Args:
            method: choose from "trained" (loading from best trained EBMs),
                and "random" (initialize from random parameters or load from ebm_dict)
            mode: concept type or relation type.
            ebm_model_type: model_type for the EBMs. Choose from "CEBM", "ConjEBM".
            ebm_dict: if not None, will contain some already-existing EBMs, whose key are the concepts/relation/operator type.
                Only effective if mode=="random".
            kwargs: init parameters for the EBM model if mode == "random".

        Returns:
            ebm_dict: A dictionary of EBMs, whose keys are the concept/relation/operator's type.
        """

        if not self.is_default_ebm and mode == DEFAULT_OBJ_TYPE:
            return

        from reasoning.experiments.models import ConceptEBM, load_model_atom
        # Initialize the parameters:
        channel_base = kwargs["channel_base"] if "channel_base" in kwargs else 128
        two_branch_mode = kwargs["two_branch_mode"] if "two_branch_mode" in kwargs else "concat"
        c_repr_mode = kwargs["c_repr_mode"] if "c_repr_mode" in kwargs else "c2"
        c_repr_first = kwargs["c_repr_first"] if "c_repr_first" in kwargs else 2
        if method == "trained":
            self.ebm_dict[mode] = load_model_atom(model_atom_str=mode, model_type=ebm_model_type, device=self.device)
        elif method == "random":
            if mode not in self.ebm_dict:
                if ebm_model_type == "CEBM":
                    if ebm_mode == "concept":
                        Dict = CONCEPTS if CONCEPTS is not None else self.CONCEPTS
                    elif ebm_mode == "operator":
                        Dict = OPERATORS if OPERATORS is not None else self.OPERATORS
                    else:
                        raise
                    if (not self.is_ebm_share_param) or (self.is_ebm_share_param and not self.ebm_dict.is_model_exist(ebm_mode)):
                        self.ebm_dict[mode] = ConceptEBM(
                            mode=ebm_mode,
                            in_channels=self.in_channels,
                            repr_dim=REPR_DIM,
                            channel_base=channel_base,
                            two_branch_mode=two_branch_mode,
                            c_repr_mode=c_repr_mode,
                            c_repr_first=c_repr_first,
                            z_mode="None" if self.is_relation_z is False and ebm_mode=="operator" else self.z_mode,
                            z_first=self.z_first,
                            z_dim=self.z_dim,
                            w_type=self.w_type,
                            mask_mode=self.mask_mode,
                            pos_embed_mode=self.pos_embed_mode,
                            aggr_mode=self.aggr_mode,
                            img_dims=self.img_dims,
                            is_spec_norm=self.is_spec_norm,
                            act_name=self.act_name,
                            normalization_type=self.normalization_type,
                        ).set_c(c_repr=Dict[mode].get_node_repr()[None], c_str=mode).to(self.device)
                    else:
                        self.ebm_dict.add_c_repr(
                            c_repr=Dict[mode].get_node_repr()[None].to(self.device),
                            c_str=mode,
                            ebm_mode=ebm_mode,
                        )
                else:
                    raise Exception("model_type '{}' is not valid!".format(model_type))
        else:
            raise Exception("method '{}' is not valid!".format(method))
        return mode


    ## EBM-related functions:
    def get_ebm(self, src):
        """Get individual EBMs from concept node or relations.

        Args:
            src: for obtaining the EBM for concept node, use node_name. For relation, use 
                a tuple of (source_node_name, target_node_name).

        Returns:
            self_ebm_dict: A dictionary of {mode: ebm_model}.
        """
        self_ebm_dict = {}
        if isinstance(src, str):
            placeholder = self.get_node_content(src)
            ebm_key = placeholder.ebm_key
            if ebm_key is not None:
                self_ebm_dict[ebm_key] = self.ebm_dict[ebm_key]
        elif isinstance(src, tuple):
            k = 0
            src = (self.get_node_name(src[0]), self.get_node_name(src[1]))
            for edge in self.edges:
                if src[0] == edge[0] and src[1] == edge[1]:
                    info = self.edges[(src[0], src[1], k)]
                    placeholder = info["value"]
                    ebm_key = placeholder.ebm_key
                    self_ebm_dict[ebm_key] = self.ebm_dict[ebm_key]
                    k += 1
        else:
            raise Exception("src must be a str (for concept) or tuple (for relation)!")
        if len(self_ebm_dict) == 0:
            self_ebm_dict = None
        return self_ebm_dict


    def get_ebms(self, ebm_type="all"):
        """Get all the EBM keys.

        Args:
            ebm_type: type of the EBMs included. Choose from "all" (all types), "concept" and "relation".

        Returns:
            self_ebm_dict_all: A dictionary with key being the src, and value being a dictionary of {mode: ebm_model}.
        """
        self_ebm_dict_all = {}
        if ebm_type in ["concept", "all"]:
            for node in self.nodes:
                self_ebm_dict = self.get_ebm(node)
                if self_ebm_dict is not None:
                    self_ebm_dict_all[node] = self_ebm_dict
        if ebm_type in ["relation", "all"]:
            for edge in self.edges:
                edge_tuple = tuple(edge[:2])
                self_ebm_dict_all[edge_tuple] = self.get_ebm(edge_tuple)
        return self_ebm_dict_all


    def get_ebm_loc(self, ebm_type="all"):
        """Obtain the dictionary of what masks each EBM refers to.
        Has the format of {ebm_key: [list of obj_names or obj_pairs]}."""
        ebm_loc_dict = {}
        if ebm_type in ["concept", "all"]:
            for node in self.nodes:
                self_ebm_dict = self.get_ebm(node)
                if self_ebm_dict is not None:
                    record_data(ebm_loc_dict, [node]*len(self_ebm_dict), list(self_ebm_dict.keys()))
        if ebm_type in ["relation", "all"]:
            for edge in self.edges:
                edge_tuple = tuple(edge[:2])
                self_ebm_dict = self.get_ebm(edge_tuple)
                record_data(ebm_loc_dict, [edge_tuple]*len(self_ebm_dict), list(self_ebm_dict.keys()))
        return ebm_loc_dict


    def get_z_mode_dict(self):
        """
        z_mode_dict: E.g.
            OrderedDict([('obj_0:c0', 'c2'),
                 ('obj_1:c1', 'c2'),
                 ('obj_2:c2', 'c2'),
                 ('obj_3:c3', 'c2'),
                 ('obj_4:c4', 'c2'),
                 ('obj_5:c5', 'c2'),
                 (('obj_6:Image', 'obj_7:Image'), 'None'),
                 (('obj_8:Image', 'obj_9:Image'), 'None'),
                 (('obj_10:Image', 'obj_11:Image'), 'None')])
        """
        ebm_dict = self.get_ebms()
        z_mode_dict = OrderedDict()
        for key in ebm_dict:
            if isinstance(key, str):
                z_mode_dict[key] = self.z_mode
            else:
                assert isinstance(key, tuple)
                z_mode_dict[key] = self.z_mode if self.is_relation_z else "None"
        return z_mode_dict


    def get_mask_info(self):
        """
        mask_info: a dictionary containing information about the masks. E.g.
            {
                id_to_type: {0: ("concept", 1), 1: ("relation", 0), 2: ("concept", 0), ...},  
                    # The number are chosen from {0,1}, which indicates the number of object slot this mask occupies, for computing mutual_exclusive loss.
                id_same_relation: [(1,3), (4,6), ...],
            }
        """
        key_to_id = {key: i for i, key in enumerate(self.nodes)}
        id_to_type = {}
        id_same_relation = []
        for ebm_key, item in self.get_ebm_loc("concept").items():
            assert len(item) == 1
            obj_slot_value = int(item[0].split(":")[-1] != DEFAULT_OBJ_TYPE)
            id_to_type[key_to_id[item[0]]] = ("concept", obj_slot_value)
        for ebm_key, item in self.get_ebm_loc("relation").items():
            assert len(item) == 1
            obj_slot_value_0 = int(item[0][0].split(":")[-1] != DEFAULT_OBJ_TYPE)
            obj_slot_value_1 = int(item[0][1].split(":")[-1] != DEFAULT_OBJ_TYPE)
            id_to_type[key_to_id[item[0][0]]] = ("relation", obj_slot_value_0)
            id_to_type[key_to_id[item[0][1]]] = ("relation", obj_slot_value_1)
            id_same_relation.append((key_to_id[item[0][0]], key_to_id[item[0][1]]))
        mask_info = {
            "id_to_type": id_to_type,
            "id_same_relation": id_same_relation,
            "z_mode_dict": self.get_z_mode_dict(),
        }
        return mask_info


    def get_masks_ebms_status(self, w_op_dict, threshold=0.5, active_n_pixel_threshold=3):
        mask_active = {}
        for key, mask in w_op_dict.items():
            is_active = ((mask>=threshold).float().sum((1,2,3)) > active_n_pixel_threshold).all().item()
            mask_active[key] = is_active
        ebm_loc_dict = self.get_ebm_loc()
        ebm_active = {}
        for ebm_key in self.ebm_dict:
            if ebm_key not in ebm_loc_dict:
                continue
            for keys in ebm_loc_dict[ebm_key]:
                # key: a mask name or tuple of mask names
                if isinstance(key, tuple):
                    is_ebm_active = mask_active[key[0]] and mask_active[key[1]]
                else:
                    is_ebm_active = mask_active[key]
                ebm_active[ebm_key] = is_ebm_active
        return mask_active, ebm_active


    def forward(self, input, w, c_repr=None, z=None, zgnn=None, wtarget=None, batch_shape=None, is_E_tensor=False):
        """
        Computes the energy given the input and the objects.

        Args:
            input:  input image, [B, C, H, W]
            w:  if the EBM has type of "CEBM", then w is a list (with length len(self.nodes)) of masks, each with shape [B, 1, H, W], where the list is ordered by self.nodes;
                if the EBM has type of "ConjEBM", then w is a list (with length len(self.nodes)) of objects, each with shape [B, channel_size, H, W]
            c_repr: embedding, , act as a placeholder
            z:  list of latent representation. If self.z_mode != "None", then it is
                a list of length len(self.get_ebms()).

        Returns:
            energy: the energy for the given inputs and objects (w).
        """
        assert len(self.nodes) == len(w)
        self.info = {}
        ebm_dict_all = self.get_ebms()
        nodes = list(self.topological_sort)
        E_all_list = []
        if self.is_selector_gnn:
            c_all_list = []
        for kk, (src, ebm_dict) in enumerate(ebm_dict_all.items()):
            if ebm_dict is None:
                continue
            if isinstance(src, str):
                # concept-EBM:
                idx = nodes.index(src)
                for mode, ebm in ebm_dict.items():
                    # The ebm_dict here is a dict of {mode: ebm}
                    z_ele = (z[kk],) if self.z_mode != "None" else None
                    energy_ele = ebm(input, (w[idx],), z=z_ele)
                    # The keys have the format of e.g. 'obj_2:c1->c1', "('obj_0:c0', 'obj_1:c1')->r0":
                    E_all_list.append(energy_ele)
                    self.info["{}->{}".format(src, mode)] = to_np_array(energy_ele)
                    if self.is_selector_gnn:
                        c_all_list.append(ebm.c_repr)
            elif isinstance(src, tuple):
                # relation-EBM:
                w_ele = tuple(w[nodes.index(src_ele)] for src_ele in src)
                for mode, ebm in ebm_dict.items():
                    z_ele = (z[kk],) if self.z_mode != "None" and z[kk] is not None else None
                    if not (z_ele is None and w_ele[0] is None and w_ele[1] is None):
                        energy_ele = ebm((input, input), w_ele, z=z_ele)
                        E_all_list.append(energy_ele)
                        self.info["{}->{}".format(src, mode)] = to_np_array(energy_ele)
                        if self.is_selector_gnn:
                            c_all_list.append(ebm.c_repr)
            else:
                raise Exception("src must be a str or a 2-tuple!")
        E_all_list = torch.stack(E_all_list)  # [n_ebms, B_task * B_example, 1]
        energy = E_all_list.sum(0)
        if self.is_selector_gnn and zgnn is not None:
            energy_gnn = self.gnn(w, z, tuple(c_all_list), zgnn, wtarget, batch_shape=batch_shape, x=input if self.z_mode=="None" else None)  # [B_task, B_example, 1]
            energy_gnn = energy_gnn.view(-1, 1)   # [B_task * B_example, 1]
            energy = energy + energy_gnn
        if is_E_tensor:
            self.info["E_all"] = E_all_list
            if self.is_selector_gnn:
                self.info["E_gnn"] = energy_gnn
        return energy


    def forward_NN_op(
        self,
        input,
        op_name,
        **kwargs
    ):
        """
        Performs the forward operation using the EBM on the node.
        """
        # E.g. op_name = "Identity-1:Image->sc$obj_0:c1", node_name = "obj_0:c1":
        node_name = op_name.split("->")[-1].split("$")[-1]
        node_mode = node_name.split(":")[-1]
        ebm = self.get_ebm(node_name)[node_mode].to(input.device)

        # Get hyperparameters:
        lambd_start = kwargs["lambd_start"] if "lambd_start" in kwargs and kwargs["lambd_start"] is not None else self.lambd_start if self.lambd_start is not None else 0.1
        lambd = kwargs["lambd"] if "lambd" in kwargs and kwargs["lambd"] is not None else self.lambd if self.lambd is not None else 0.005
        image_value_range = kwargs["image_value_range"] if "image_value_range" in kwargs and kwargs["image_value_range"] is not None else self.image_value_range if self.image_value_range is not None else "0,1"
        step_size_start = kwargs["step_size_start"] if "step_size_start" in kwargs else -1
        step_size = kwargs["step_size"] if "step_size" in kwargs and kwargs["step_size"] is not None else self.step_size if self.step_size is not None else 20
        step_size_img = kwargs["step_size_img"] if "step_size_img" in kwargs and kwargs["step_size_img"] is not None else self.step_size_img if self.step_size_img is not None else -1
        step_size_zgnn = kwargs["step_size_zgnn"] if "step_size_zgnn" in kwargs and kwargs["step_size_zgnn"] is not None else self.step_size_zgnn if self.step_size_zgnn is not None else 2
        step_size_wtarget = kwargs["step_size_wtarget"] if "step_size_wtarget" in kwargs and kwargs["step_size_wtarget"] is not None else self.step_size_wtarget if self.step_size_wtarget is not None else -1
        SGLD_is_anneal = kwargs["SGLD_is_anneal"] if "SGLD_is_anneal" in kwargs and kwargs["SGLD_is_anneal"] is not None else self.SGLD_is_anneal if self.SGLD_is_anneal is not None else True
        SGLD_is_penalize_lower = kwargs["SGLD_is_penalize_lower"] if "SGLD_is_penalize_lower" in kwargs and kwargs["SGLD_is_penalize_lower"] is not None else self.SGLD_is_penalize_lower if self.SGLD_is_penalize_lower is not None else True
        SGLD_mutual_exclusive_coef = kwargs["SGLD_mutual_exclusive_coef"] if "SGLD_mutual_exclusive_coef" in kwargs and kwargs["SGLD_mutual_exclusive_coef"] is not None else self.SGLD_mutual_exclusive_coef if self.SGLD_mutual_exclusive_coef is not None else 0
        SGLD_pixel_entropy_coef = kwargs["SGLD_pixel_entropy_coef"] if "SGLD_pixel_entropy_coef" in kwargs and kwargs["SGLD_pixel_entropy_coef"] is not None else self.SGLD_pixel_entropy_coef if self.SGLD_pixel_entropy_coef is not None else 0
        SGLD_pixel_gm_coef = kwargs["SGLD_pixel_gm_coef"] if "SGLD_pixel_gm_coef" in kwargs and kwargs["SGLD_pixel_gm_coef"] is not None else self.SGLD_pixel_gm_coef if self.SGLD_pixel_gm_coef is not None else 0
        SGLD_object_exceed_coef = kwargs["SGLD_object_exceed_coef"] if "SGLD_object_exceed_coef" in kwargs else 0
        # For selector discovery:
        SGLD_iou_batch_consistency_coef = kwargs["SGLD_iou_batch_consistency_coef"] if "SGLD_iou_batch_consistency_coef" in kwargs and kwargs["SGLD_iou_batch_consistency_coef"] is not None else self.SGLD_iou_batch_consistency_coef if self.SGLD_iou_batch_consistency_coef is not None else 0
        SGLD_iou_concept_repel_coef = kwargs["SGLD_iou_concept_repel_coef"] if "SGLD_iou_concept_repel_coef" in kwargs and kwargs["SGLD_iou_concept_repel_coef"] is not None else self.SGLD_iou_concept_repel_coef if self.SGLD_iou_concept_repel_coef is not None else 0
        SGLD_iou_relation_repel_coef = kwargs["SGLD_iou_relation_repel_coef"] if "SGLD_iou_relation_repel_coef" in kwargs and kwargs["SGLD_iou_relation_repel_coef"] is not None else self.SGLD_iou_relation_repel_coef if self.SGLD_iou_relation_repel_coef is not None else 0
        SGLD_iou_relation_overlap_coef = kwargs["SGLD_iou_relation_overlap_coef"] if "SGLD_iou_relation_overlap_coef" in kwargs and kwargs["SGLD_iou_relation_overlap_coef"] is not None else self.SGLD_iou_relation_overlap_coef if self.SGLD_iou_relation_overlap_coef is not None else 0
        SGLD_iou_attract_coef = kwargs["SGLD_iou_attract_coef"] if "SGLD_iou_attract_coef" in kwargs and kwargs["SGLD_iou_attract_coef"] is not None else self.SGLD_iou_attract_coef if self.SGLD_iou_attract_coef is not None else 0

        # Other settings:
        sample_step = kwargs["sample_step"] if "sample_step" in kwargs else 60
        ensemble_size = kwargs["ensemble_size"] if "ensemble_size" in kwargs else 1
        w_type = kwargs["w_type"] if "w_type" in kwargs else "image+mask"
        w_init_type = kwargs["w_init_type"] if "w_init_type" in kwargs and kwargs["w_init_type"] is not None else self.w_init_type if self.w_init_type is not None else "random"
        assert w_init_type in ["random", "input", "input-mask", "input-gaus"] or w_init_type.startswith("k-means")

        args = get_pdict()(
            lambd_start=lambd_start,
            lambd=lambd,
            image_value_range=image_value_range,
            step_size_start=step_size_start,
            step_size=step_size,
            step_size_img=step_size_img,
            step_size_zgnn=step_size_zgnn,
            step_size_wtarget=step_size_wtarget,
            image_size=input.shape[-2:],
            w_type=w_type,
            SGLD_is_anneal=SGLD_is_anneal,
            SGLD_is_penalize_lower=SGLD_is_penalize_lower,
            SGLD_object_exceed_coef=SGLD_object_exceed_coef,
            SGLD_mutual_exclusive_coef=SGLD_mutual_exclusive_coef,
            SGLD_pixel_entropy_coef=SGLD_pixel_entropy_coef,
            SGLD_pixel_gm_coef=SGLD_pixel_gm_coef,
            # Selector:
            SGLD_iou_batch_consistency_coef=SGLD_iou_batch_consistency_coef,
            SGLD_iou_concept_repel_coef=SGLD_iou_concept_repel_coef,
            SGLD_iou_relation_repel_coef=SGLD_iou_relation_repel_coef,
            SGLD_iou_relation_overlap_coef=SGLD_iou_relation_overlap_coef,
            SGLD_iou_attract_coef=SGLD_iou_attract_coef,
        )
        with torch.enable_grad():
            (img_ensemble_sorted, neg_mask_ensemble_sorted, z_ensemble_sorted, _, _), neg_out_ensemble_sorted = ebm.ground(
                input,
                args=args,
                ensemble_size=ensemble_size,
                topk=-1,
                w_init_type=w_init_type,
                sample_step=60,
                isplot=False,
            )
        # Using the w with the lowest energy in the ensemble:
        # For now, repeat the mask channel 10 times to mimic the full object:
        w_grounded = neg_mask_ensemble_sorted[0][:, 0]
        return w_grounded


    def forward_NN(
        self,
        input,
        **kwargs
    ):
        """
        Performs the forward operation for the full selector with EBM.

        Two scenarios:
        (1) "obj" in w_type:
            The 1st SGLD can obtain the w_op_dict

        (2) "mask" in w_type and z_mode != "None":
            The 1st SGLD obtain the mask and z, and
            the 2nd SGLD reconstruct the img based on the inferred mask and z.

        Returns:
            pred: For (1), will simply be combination from w_op_dict.
                    For (2), will be reconstruction based on the mask from 1st SGLD and the reconstructed img from the 2nd SGLD.
                    The recons has the same shape as the input.
            w_op_dict: will always be the mask/obj from the 1st SGLD
        """
        def squeeze_batch(tensor, is_squeeze):
            if is_squeeze:
                tensor = tensor.squeeze(0)
            return tensor

        inp_hash = persist_hash(str(input))
        if self.cache_forward and inp_hash in self.forward_cache:
            return self.forward_cache[inp_hash]

        # If there is an additional task-batch dimension, combine that with the example-batch:
        if len(input.shape) == 5:
            batch_dims = input.shape[:2]
            batch_shape = tuple(batch_dims)
        else:
            assert len(input.shape) == 4
            batch_dims = input.shape[:1]
            batch_shape = None
        input = input.view(-1, *input.shape[-3:])

        # Setting up the hyperparameters:
        lambd_start = kwargs["lambd_start"] if "lambd_start" in kwargs and kwargs["lambd_start"] is not None else self.lambd_start if self.lambd_start is not None else 0.1
        lambd = kwargs["lambd"] if "lambd" in kwargs and kwargs["lambd"] is not None else self.lambd if self.lambd is not None else 0.005
        image_value_range = kwargs["image_value_range"] if "image_value_range" in kwargs and kwargs["image_value_range"] is not None else self.image_value_range if self.image_value_range is not None else "0,1"
        SGLD_is_anneal = kwargs["SGLD_is_anneal"] if "SGLD_is_anneal" in kwargs and kwargs["SGLD_is_anneal"] is not None else self.SGLD_is_anneal if self.SGLD_is_anneal is not None else True
        SGLD_is_penalize_lower = kwargs["SGLD_is_penalize_lower"] if "SGLD_is_penalize_lower" in kwargs and kwargs["SGLD_is_penalize_lower"] is not None else self.SGLD_is_penalize_lower if self.SGLD_is_penalize_lower is not None else True
        SGLD_mutual_exclusive_coef = kwargs["SGLD_mutual_exclusive_coef"] if "SGLD_mutual_exclusive_coef" in kwargs and kwargs["SGLD_mutual_exclusive_coef"] is not None else self.SGLD_mutual_exclusive_coef if self.SGLD_mutual_exclusive_coef is not None else 0
        SGLD_pixel_entropy_coef = kwargs["SGLD_pixel_entropy_coef"] if "SGLD_pixel_entropy_coef" in kwargs and kwargs["SGLD_pixel_entropy_coef"] is not None else self.SGLD_pixel_entropy_coef if self.SGLD_pixel_entropy_coef is not None else 0
        SGLD_pixel_gm_coef = kwargs["SGLD_pixel_gm_coef"] if "SGLD_pixel_gm_coef" in kwargs and kwargs["SGLD_pixel_gm_coef"] is not None else self.SGLD_pixel_gm_coef if self.SGLD_pixel_gm_coef is not None else 0
        SGLD_object_exceed_coef = kwargs["SGLD_object_exceed_coef"] if "SGLD_object_exceed_coef" in kwargs else 0
        # For selector discovery:
        SGLD_iou_batch_consistency_coef = kwargs["SGLD_iou_batch_consistency_coef"] if "SGLD_iou_batch_consistency_coef" in kwargs and kwargs["SGLD_iou_batch_consistency_coef"] is not None else self.SGLD_iou_batch_consistency_coef if self.SGLD_iou_batch_consistency_coef is not None else 0
        SGLD_iou_concept_repel_coef = kwargs["SGLD_iou_concept_repel_coef"] if "SGLD_iou_concept_repel_coef" in kwargs and kwargs["SGLD_iou_concept_repel_coef"] is not None else self.SGLD_iou_concept_repel_coef if self.SGLD_iou_concept_repel_coef is not None else 0
        SGLD_iou_relation_repel_coef = kwargs["SGLD_iou_relation_repel_coef"] if "SGLD_iou_relation_repel_coef" in kwargs and kwargs["SGLD_iou_relation_repel_coef"] is not None else self.SGLD_iou_relation_repel_coef if self.SGLD_iou_relation_repel_coef is not None else 0
        SGLD_iou_relation_overlap_coef = kwargs["SGLD_iou_relation_overlap_coef"] if "SGLD_iou_relation_overlap_coef" in kwargs and kwargs["SGLD_iou_relation_overlap_coef"] is not None else self.SGLD_iou_relation_overlap_coef if self.SGLD_iou_relation_overlap_coef is not None else 0
        SGLD_iou_attract_coef = kwargs["SGLD_iou_attract_coef"] if "SGLD_iou_attract_coef" in kwargs and kwargs["SGLD_iou_attract_coef"] is not None else self.SGLD_iou_attract_coef if self.SGLD_iou_attract_coef is not None else 0
        # Other settings:
        step_size_start = kwargs["step_size_start"] if "step_size_start" in kwargs else -1
        step_size = kwargs["step_size"] if "step_size" in kwargs and kwargs["step_size"] is not None else self.step_size if self.step_size is not None else 20
        step_size_img = kwargs["step_size_img"] if "step_size_img" in kwargs and kwargs["step_size_img"] is not None else self.step_size_img if self.step_size_img is not None else -1
        step_size_z = kwargs["step_size_z"] if "step_size_z" in kwargs and kwargs["step_size_z"] is not None else self.step_size_z if self.step_size_z is not None else 2
        step_size_zgnn = kwargs["step_size_zgnn"] if "step_size_zgnn" in kwargs and kwargs["step_size_zgnn"] is not None else self.step_size_zgnn if self.step_size_zgnn is not None else 2
        step_size_wtarget = kwargs["step_size_wtarget"] if "step_size_wtarget" in kwargs and kwargs["step_size_wtarget"] is not None else self.step_size_wtarget if self.step_size_wtarget is not None else -1
        sample_step = kwargs["sample_step"] if "sample_step" in kwargs else 60
        ensemble_size = kwargs["ensemble_size"] if "ensemble_size" in kwargs else 1
        kl_all_step = kwargs["kl_all_step"] if "kl_all_step" in kwargs else True
        is_grad = kwargs["is_grad"] if "is_grad" in kwargs else False
        w_init_type = kwargs["w_init_type"] if "w_init_type" in kwargs and kwargs["w_init_type"] is not None else self.w_init_type if self.w_init_type is not None else "random"
        indiv_sample = kwargs["indiv_sample"] if "indiv_sample" in kwargs and kwargs["indiv_sample"] is not None else self.indiv_sample if self.indiv_sample is not None else -1
        img_init_type = kwargs["img_init_type"] if "img_init_type" in kwargs else "random"
        is_return_E = kwargs["is_return_E"] if "is_return_E" in kwargs else False
        isplot = kwargs["isplot"] if "isplot" in kwargs else 0
        verbose = kwargs["verbose"] if "verbose" in kwargs else 0

        args = get_pdict()(
            lambd_start=lambd_start,
            lambd=lambd,
            image_value_range=image_value_range,
            step_size_start=step_size_start,
            step_size=step_size,
            step_size_img=step_size_img,
            step_size_z=step_size_z,
            step_size_zgnn=step_size_zgnn,
            step_size_wtarget=step_size_wtarget,
            image_size=input.shape[-2:],
            kl_all_step=kl_all_step,
            SGLD_is_anneal=SGLD_is_anneal,
            SGLD_is_penalize_lower=SGLD_is_penalize_lower,
            SGLD_object_exceed_coef=SGLD_object_exceed_coef,
            SGLD_mutual_exclusive_coef=SGLD_mutual_exclusive_coef,
            SGLD_pixel_entropy_coef=SGLD_pixel_entropy_coef,
            SGLD_pixel_gm_coef=SGLD_pixel_gm_coef,
            # Selector:
            SGLD_iou_batch_consistency_coef=SGLD_iou_batch_consistency_coef,
            SGLD_iou_concept_repel_coef=SGLD_iou_concept_repel_coef,
            SGLD_iou_relation_repel_coef=SGLD_iou_relation_repel_coef,
            SGLD_iou_relation_overlap_coef=SGLD_iou_relation_overlap_coef,
            SGLD_iou_attract_coef=SGLD_iou_attract_coef,
        )

        ebm_0 = self.ebm_dict[next(iter(self.ebm_dict))]
        if w_init_type in ["random", "input-mask", "input-gaus"] or w_init_type.startswith("k-means"):
            mask = None
        elif w_init_type == "input":
            mask_arity = len(self.nodes)
            if "obj" in ebm_0.w_type:
                assert "mask" not in ebm_0.w_type
                mask = tuple(input.detach().clone().repeat_interleave(ensemble_size, dim=0) for k in range(mask_arity))
            elif "mask" in ebm_0.w_type:
                assert "obj" not in ebm_0.w_type
                if input.shape[1] == 10:
                    mask = tuple((input[:,:1]!=1).detach().clone().repeat_interleave(ensemble_size, dim=0) for k in range(mask_arity))
                elif input.shape[1] == 3:
                    mask = None
                else:
                    raise
            else:
                raise
            for k in range(mask_arity):
                if mask is not None:
                    mask[k].requires_grad = True
        else:
            raise Exception("w_init_type '{}' is not valid!".format(w_init_type))

        is_reconstruct = "mask" in ebm_0.w_type and ebm_0.z_mode != "None"

        with torch.enable_grad():
            args.ebm_target = "mask" if ebm_0.z_mode == "None" else "mask+z"
            if self.is_selector_gnn:
                args.ebm_target += "+zgnn"
            # SGLD w.r.t. w (and z):
            (_, neg_mask_ensemble_sorted, z_ensemble_sorted, zgnn_ensemble_sorted, _), neg_out_ensemble_sorted, info = self.ground(
                input,
                args,
                mask=mask,
                ensemble_size=ensemble_size,
                topk=1,
                w_init_type=w_init_type,
                sample_step=sample_step,
                is_grad=is_grad,
                is_return_E=is_return_E,
                batch_shape=batch_shape,
                isplot=isplot,
            )
            if is_return_E:
                E_all = info["E_all"]
            
            if verbose >= 1 and self.is_selector_gnn:
                if hasattr(self.gnn, "softmax_coef"):
                    print("softmax_coef:")
                    print(self.gnn.softmax_coef)
                print(list(self.gnn.parameters())[:4])
                print("z_node:")
                if zgnn_ensemble_sorted[0] is None:
                    print(None)
                else:
                    print(zgnn_ensemble_sorted[0].squeeze())
                print("z_edge:")
                print(zgnn_ensemble_sorted[1].squeeze())
                print()

            # Obtain the w_selected (list of w selected by the refer nodes) and a dict of all ws:
            is_squeeze = True if len(input.shape) == 3 else False
            w_op_dict = {}
            z_op_dict = {}
            w_selector = []
            nodes = list(self.nodes)
            for i, neg_mask_ele in enumerate(neg_mask_ensemble_sorted):
                neg_mask_top = squeeze_batch(neg_mask_ele[:,0], is_squeeze)
                if self.refer_node_names is None or nodes[i] in self.refer_node_names:
                    # Accumulate the objects specified by the self.refer_node_names:
                    w_selector.append(neg_mask_top)
                w_op_dict[nodes[i]] = neg_mask_top
                if indiv_sample != -1:
                    z_op_dict[nodes[i]] = z_ensemble_sorted[i][:,0]

            if not is_reconstruct:
                # If not is_reconstruct, then the pred is certain combination of the w_op_dict from the first SGLD:
                assert len(w_selector[0].shape) == 4
#                 assert "obj" in ebm_0.w_type
                if w_selector[0].shape[-3] == 10:
                    # Obj with 10 color channels (first is empty channel):
                    w_selector_stack = torch.stack(w_selector)  # [n_obj, B, 10, H, W]
                    pred = torch.cat([
                        w_selector_stack[:,:,:1].mean(0),
                        w_selector_stack[:,:,1:].sum(0),
                    ], 1)
                elif w_selector[0].shape[-3] == 3:
                    # Obj with RGB channels:
                    pred = torch.stack(w_selector).sum(0)
                elif w_selector[0].shape[-3] == 1:
                    pred = None
                else:
                    raise
            else:
                # If is_reconstruct, perform a second SGLD w.r.t. img given the inferred w (and z):
                if img_init_type == "random":
                    img_value_min, img_value_max = image_value_range.split(",")
                    img_value_min, img_value_max = eval(img_value_min), eval(img_value_max)
                    img_init = torch.rand(input.shape, device=input.device) * (img_value_max - img_value_min) + img_value_min
                elif img_init_type == "input":
                    img_init = input.detach()
                else:
                    raise

                # Specify the mask according to the refer nodes:
                neg_mask_latent = []
                mask_info = self.get_mask_info()
                for i, neg_mask_ele in enumerate(neg_mask_ensemble_sorted):
                    if self.refer_node_names is None or nodes[i] in self.refer_node_names:
                        if mask_info["id_to_type"][i][0] == "relation" and self.is_relation_z is False:
                            neg_mask_latent.append(None)
                        else:
                            neg_mask_latent.append(neg_mask_ele[:,0])
                    else:
                        # If not refer nodes, then turn off the mask:
                        neg_mask_latent.append(torch.zeros(neg_mask_ele[:,0].shape).to(device))
                # 2nd SGLD w.r.t. img, given mask and z:
                z_latent = tuple(ele[:,0] if ele is not None else None for ele in z_ensemble_sorted) if "z" in args.ebm_target else None
                zgnn_latent = tuple(zgnn_ele[:,0] if zgnn_ele is not None else None for zgnn_ele in zgnn_ensemble_sorted) if zgnn_ensemble_sorted is not None else None
                args.ebm_target = "image"
                if indiv_sample != -1:
                    # Go through each EBM
                    pred = None
                    ebm_dict = self.get_ebms()
                    for key in w_op_dict.keys():
                        model = list(ebm_dict[key].values())[0]
                        (img_ensemble_sorted, _, _, _, _), neg_out_ensemble_sorted = model.ground(
                            img_init,
                            args,
                            mask=tuple([w_op_dict[key]]),
                            z=tuple([z_op_dict[key]]),
                            zgnn=zgnn_latent,
                            ensemble_size=ensemble_size,
                            topk=1,
                            sample_step=indiv_sample,
                            is_grad=is_grad,
                            batch_shape=batch_shape,
                            isplot=isplot,
                        )
                        if pred is None:
                            pred = img_ensemble_sorted[:, 0] * w_op_dict[key]
                        else:
                            pred = pred + img_ensemble_sorted[:, 0] * w_op_dict[key]
                    img_value_min, img_value_max = args.image_value_range.split(",")
                    img_value_min, img_value_max = eval(img_value_min), eval(img_value_max)
                    pred = pred.clamp(min=img_value_min, max=img_value_max)
                else:
                    (img_ensemble_sorted, _, _, _, _), neg_out_ensemble_sorted, info = self.ground(
                        img_init,
                        args,
                        mask=neg_mask_latent,
                        z=z_latent,
                        zgnn=zgnn_latent,
                        ensemble_size=ensemble_size,
                        topk=1,
                        sample_step=sample_step,
                        is_grad=is_grad,
                        batch_shape=batch_shape,
                        isplot=isplot,
                    )
                    pred = img_ensemble_sorted[:,0]
        if pred is not None:
            assert pred.shape == input.shape
            # Recover the task-batch dimension:
            pred = pred.view(*batch_dims, *pred.shape[-3:])
        w_op_dict = {key: item.view(*batch_dims, *item.shape[-3:]) for key, item in w_op_dict.items()}

        # Store the results in a cache, to be used by the policy:
        if self.cache_forward:
            # Input should be a batch of examples
            self.forward_cache[inp_hash] = (pred, w_op_dict)
        if is_return_E:
            return pred, w_op_dict, E_all
        else:
            return pred, w_op_dict


    def ground(
        self,
        input,
        args,
        mask=None,
        c_repr=None,
        z=None,
        zgnn=None,
        wtarget=None,
        ensemble_size=18,
        topk=-1,
        w_init_type="random",
        sample_step=150,
        ground_truth_mask=None,
        is_grad=False,
        is_return_E=False,
        batch_shape=None,
        isplot=2,
        **kwargs
    ):
        """
        Ground the input with the selector itself and return the discovered objects (masks).
        """
        def init_neg_mask(input, init, ensemble_size, mask_arity, w_type):
            """Initialize negative mask"""
            device = input.device
            w_dim = input.shape[1] if "obj" in w_type else 1
            neg_mask = tuple(torch.rand(input.shape[0]*ensemble_size, w_dim, *input.shape[2:]).to(device) for k in range(mask_arity))
            input_l = repeat_n(input, n_repeats=ensemble_size)
            if init == "input-mask":
                assert input.shape[1] == 10
                neg_mask = tuple(neg_mask[k] * (input_l.argmax(1)[:, None] != 0) for k in range(mask_arity))
            elif init == "input-gaus":
                means = input_l.argmax(1)[:, None].float()
                std = 0.01 * torch.ones_like(means, device=device)
                neg_mask = tuple(neg_mask[k] * torch.normal(means, std).clamp(min=0, max=1) for k in range(mask_arity))
            elif init.startswith("k-means"):
                parts = init.split("^")
                num_clusters = mask_arity if len(parts) < 2 else eval(parts[1])
                input_l_dim = input_l.shape
                # Flatten H, W then permute to get [batch_size, n_pixels, n_channels]. Flatten to
                # [batch_size * n_pixels, n_channels]
                input_flat = input_l.flatten(2).permute(0, 2, 1).flatten(end_dim=1)
                cluster_ids, cluster_centers = kmeans(
                    X=input_flat, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=False,
                )
                # Get the most common cluster
                background_cluster, _ = cluster_ids.mode(0)
                all_ex_mask = (cluster_ids != background_cluster).reshape(input_l_dim[0], *input_l_dim[2:]).unsqueeze(1).to(device)
                neg_mask = tuple(neg_mask[k] * all_ex_mask for k in range(mask_arity))
            for k in range(mask_arity):
                neg_mask[k].requires_grad = True
            return neg_mask

        def plot_discovered_mask_summary(num_examples: int, neg_mask_ensemble_sorted: torch.Tensor, should_quantize: bool):
            plt.figure(figsize=(18,3))
            for batch_idx in range(len(neg_mask_ensemble_sorted[0])):
                for ex in range(num_examples):
                    for mask_idx in range(len(neg_mask_ensemble_sorted)):
                        ax = plt.subplot(1, num_examples, ex + 1)
                        # Pull single-channel (0)
                        mask = to_np_array(neg_mask_ensemble_sorted[mask_idx][batch_idx][ex][0])
                        image = np.zeros((*mask.shape, 4)) # (H, W, C, alpha)
                        color = np.asarray(matplotlib.colors.to_rgb(COLOR_LIST[mask_idx]))
                        for h in range(mask.shape[0]):
                            for w in range(mask.shape[1]):
                                opacity = mask[h][w].round() if should_quantize else mask[h][w]
                                pixel = opacity * np.asarray((*color, 0.5)) # add alpha channel
                                image[h][w] = pixel
                        plt.imshow(image)
                        ax.set_title("E: {:.5f}\n".format(neg_out_ensemble_sorted[batch_idx][ex]))
            plt.show()

        from reasoning.experiments.models import neg_mask_sgd_ensemble
        ebm_0 = self.ebm_dict[next(iter(self.ebm_dict))]
        z_mode = ebm_0.z_mode

        # Update args:
        args = deepcopy(args)
        for key, value in kwargs.items():
            setattr(args, key, value)
        args.sample_step = sample_step
        args.is_two_branch = True if ebm_0.mode in ["operator"] else False

        assert (not isinstance(input, tuple)) and (not isinstance(input, list))
        if input is not None:
            if len(input.shape) == 3:
                input = input[None]
            assert len(input.shape) == 4
            device = input.device
            args.is_image_tuple = (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 2
        else:
            device = mask[0].device
            args.is_image_tuple = False

        if ground_truth_mask is not None:
            print("Ground truth mask energies:")
            energy = self.forward(input, ground_truth_mask, c_repr)
            # Plot all masks together, followed by each individual mask
            visualize_matrices(torch.cat([sum(ground_truth_mask), *ground_truth_mask], dim=0).squeeze(1),
                               images_per_row=len(ground_truth_mask) + 1,
                               subtitles=[
                                   # Print overall energy
                                   "E: {:.5f}\n".format(float(energy)) + \
                                    # Print energy of individual components
                                    "\n".join(["{}: {:.5f}".format(key, float(self.info[key])) for key in self.info])] + [""] * len(ground_truth_mask))


        # Initialization:
        if input is not None:
            if mask is None:
                neg_mask = init_neg_mask(input, init=w_init_type, ensemble_size=ensemble_size, mask_arity=self.mask_arity, w_type=ebm_0.w_type)
            else:
                neg_mask = tuple(repeat_n(mask[k], n_repeats=ensemble_size) for k in range(len(mask)))
            if z_mode != "None":
                z_mode_dict = self.get_z_mode_dict()
                z_tuple = []
                for k, (ebm_key, z_mode_ele) in enumerate(z_mode_dict.items()):
                    if z_mode_ele == "None":
                        z_tuple.append(None)
                    else:
                        if z is None:
                            z_tuple.append(torch.rand(neg_mask[0].shape[0], ebm_0.z_dim, device=device))
                        else:
                            z_tuple.append(repeat_n(z[k], n_repeats=ensemble_size))
                z = tuple(z_tuple)
            else:
                z = None
        else:
            neg_mask = tuple(repeat_n(mask[k], n_repeats=ensemble_size) for k in range(len(mask)))
        # Perform SGLD:
        (img_ensemble, neg_mask_ensemble, z_ensemble, zgnn_ensemble, wtarget_ensemble), neg_out_list_ensemble, info_ensemble = neg_mask_sgd_ensemble(
            self, input, neg_mask, c_repr, z=z, zgnn=zgnn, wtarget=wtarget,
            args=args,
            ensemble_size=ensemble_size,
            mask_info = self.get_mask_info(),
            is_grad=is_grad,
            is_return_E=is_return_E,
            batch_shape=batch_shape,
        )  # neg_mask_ensemble: [ensemble_size, B, C, H, W]; neg_out_list_ensemble: [sample_step, ensemble_size, B]
        neg_out_ensemble = neg_out_list_ensemble[-1]  # neg_out_ensemble: [ensemble_size, B]

        topk_core = ensemble_size if topk==-1 else min(ensemble_size, topk)
        # Sort the obtained results by energy for each example:
        neg_out_ensemble = torch.FloatTensor(neg_out_ensemble).transpose(0,1)  # neg_out_ensemble (new): [B, ensemble_size]
        neg_out_argsort = neg_out_ensemble.argsort(1)  # [B, ensemble_size]
        batch_size = neg_out_argsort.shape[0]
        neg_out_ensemble_sorted = torch.stack([neg_out_ensemble[i][neg_out_argsort[i]][:topk_core] for i in range(batch_size)])  # [B, ensemble_size]
        if zgnn_ensemble is not None or wtarget is not None:
            neg_task_out_ensemble = neg_out_ensemble.reshape(*batch_shape, -1).mean(1)  # [B_task, ensemble_size]
            neg_task_out_argsort = neg_task_out_ensemble.argsort(1)

        if img_ensemble is not None:
            if args.is_image_tuple:
                img_ensemble = tuple(img_ensemble[k].transpose(0,1) for k in range(len(img)))  # Each element [B, ensemble_size, C, H, W]
                img_ensemble_sorted = []
                for k in range(len(img)):
                    img_ensemble_sorted.append(torch.stack([img_ensemble[k][i][neg_out_argsort[i]][:topk_core] for i in range(batch_size)]))
                img_ensemble_sorted = tuple(img_ensemble_sorted)  # each element: [B, ensemble_size, C, H, W] sorted along dim=1 according to neg_out
            else:
                img_ensemble = img_ensemble.transpose(0,1)  # [B, ensemble_size, C, H, W]
                img_ensemble_sorted = torch.stack([img_ensemble[i][neg_out_argsort[i]][:topk_core] for i in range(batch_size)])
        else:
            img_ensemble_sorted = None

        if neg_mask_ensemble is not None:
            neg_mask_ensemble = tuple(neg_mask_ensemble[k].transpose(0,1) for k in range(self.mask_arity))  # Each element [B, ensemble_size, C, H, W]
            neg_mask_ensemble_sorted = []
            for k in range(self.mask_arity):
                neg_mask_ensemble_sorted.append(torch.stack([neg_mask_ensemble[k][i][neg_out_argsort[i]][:topk_core] for i in range(len(neg_mask_ensemble[0]))]))
            neg_mask_ensemble_sorted = tuple(neg_mask_ensemble_sorted)  # each element: [B, ensemble_size, C, H, W] sorted along dim=1 according to neg_out
        else:
            neg_mask_ensemble_sorted = None

        if z_ensemble is not None:
            z_ensemble = tuple(z_ensemble[k].transpose(0,1) if z_ensemble[k] is not None else None for k in range(len(z_ensemble)))  # Each element [B, ensemble_size, Z]
            z_ensemble_sorted = []
            for k in range(len(z_ensemble)):
                if z_ensemble[k] is not None:
                    z_ensemble_sorted.append(torch.stack([z_ensemble[k][i][neg_out_argsort[i]][:topk_core] for i in range(batch_size)]))
                else:
                    z_ensemble_sorted.append(None)
            z_ensemble_sorted = tuple(z_ensemble_sorted)  # each element: [B, ensemble_size, Z] sorted along dim=1 according to neg_out
        else:
            z_ensemble_sorted = None

        if zgnn_ensemble is not None:
            zgnn_ensemble = tuple(zgnn_ensemble[k].transpose(0,1) if zgnn_ensemble[k] is not None else None for k in range(len(zgnn_ensemble)))  # Each element [B, ensemble_size, Zgnn]
            zgnn_ensemble_sorted = []
            for k in range(len(zgnn_ensemble)):
                if zgnn_ensemble[k] is not None:
                    zgnn_ensemble_sorted.append(torch.stack([zgnn_ensemble[k][i][neg_task_out_argsort[i]][:topk_core] for i in range(batch_shape[0])]))
                else:
                    zgnn_ensemble_sorted.append(None)
            zgnn_ensemble_sorted = tuple(zgnn_ensemble_sorted)  # each element: [B, ensemble_size, zgnn_dim] sorted along dim=1 according to neg_out
        else:
            zgnn_ensemble_sorted = None

        if wtarget_ensemble is not None:
            wtarget_ensemble = wtarget_ensemble.transpose(0,1)  # [B, ensemble_size, w_dim, H, W]
            wtarget_ensemble_sorted = torch.stack([wtarget_ensemble[i][neg_out_argsort[i]][:topk_core] for i in range(batch_size)])
        else:
            wtarget_ensemble_sorted = None

        # Obtain each individual energy for component models:
        info = {}
        if is_return_E:
            info["E_all"] = info_ensemble.pop("E_all")
            self.info.pop("E_all", None)
        neg_out_argsort = to_np_array(neg_out_argsort)
        for key, value in self.info.items():
            value_reshape = value.reshape(ensemble_size, -1).T  # [B, ensemble_size]
            info[key] = []
            for i in range(len(value_reshape)):
                info[key].append(value_reshape[i][neg_out_argsort[i]][:topk_core])
            info[key] = np.stack(info[key])

        NUM_PREV_EXAMPLES = min(6, ensemble_size)

        if isplot >= 2:
            # Plot SGLD learning curve:
            print("SGLD learning curve:")
            plt.figure(figsize=(12,6))
            for i in range(min(neg_out_list_ensemble.shape[-1], 6)):  # neg_out_list_ensemble: [sample_step, ensemble_size, B]
#                 print("Example {}".format(i))
                for k in range(min(6, ensemble_size)):
                    plt.plot(neg_out_list_ensemble[:, neg_out_argsort[i][k],i], c=COLOR_LIST[k], label="id_{}".format(k), alpha=0.4)
            plt.legend()
            plt.show()
        if isplot >= 3 and "mask" in args.ebm_target:
            import matplotlib
            w_type = ebm_0.w_type
            # Show original input image for reference
            print("Original inputs:")
            for i in range(len(neg_mask_ensemble_sorted)):
                visualize_matrices(input[i:i+1].argmax(1).repeat_interleave(NUM_PREV_EXAMPLES, 0))

            # Plot a summary plot, superimposing different masks such that each mask has a different color
            if "mask" in w_type:
                print(f"Top-{NUM_PREV_EXAMPLES} lowest-energy mask sets, all plotted together")
                print(f"Key parameters: SGLD_mutual_exclusive_coef={str(args.SGLD_mutual_exclusive_coef)}",
                        f"object_exceed_coef={str(args.SGLD_object_exceed_coef)}")
                plot_discovered_mask_summary(NUM_PREV_EXAMPLES, neg_mask_ensemble_sorted, should_quantize=False)

                # Plot the same plot, but this time quantized so there are no color gradations
                print("Quantized plot:")
                plot_discovered_mask_summary(NUM_PREV_EXAMPLES, neg_mask_ensemble_sorted, should_quantize=True)

            # For each batch element
            for i in range(len(neg_mask_ensemble_sorted[0])):
                print("Example {}".format(i))
                # Loop through each mask (show them horizontally)
                for k in range(len(neg_mask_ensemble_sorted)):
                    if "mask" in w_type:
                        img_to_plot = torch.round(neg_mask_ensemble_sorted[k][i][:NUM_PREV_EXAMPLES].squeeze(1))
                    elif "obj" in w_type:
                        img_to_plot = neg_mask_ensemble_sorted[k][i][:NUM_PREV_EXAMPLES].argmax(1)
                    else:
                        raise
                    visualize_matrices(
                        img_to_plot, images_per_row=NUM_PREV_EXAMPLES,
                        subtitles=["E: {:.5f}\n".format(neg_out_ensemble_sorted[i][j]) + "\n".join(["{}: {:.5f}".format(key, info[key][i][j]) for key in info]) for j in range(NUM_PREV_EXAMPLES)] if k == 0 else None
                    )
        return (img_ensemble_sorted, neg_mask_ensemble_sorted, z_ensemble_sorted, zgnn_ensemble_sorted, wtarget_ensemble_sorted), neg_out_ensemble_sorted, info


    def forward_NN_relation(
        self,
        train_input,
        train_target,
        test_input,
        **kwargs
    ):
        def get_compat_ids(mask_info):
            def get_triu_ids(array, is_triu=True):
                if isinstance(array, Number):
                    array = np.arange(array)
                rows_matrix, col_matrix = np.meshgrid(array, array)
                matrix_cat = np.stack([rows_matrix, col_matrix], -1)
                rr, cc = np.triu_indices(len(matrix_cat), k=1)
                rows, cols = matrix_cat[cc, rr].T
                return rows, cols

            n_masks = len(mask_info['id_to_type'])
            concept_ids = [id for id, item in mask_info["id_to_type"].items() if item[0] == "concept"]
            concept_rows, concept_cols = get_triu_ids(concept_ids)
            if len(mask_info["id_same_relation"]) > 0:
                relation_rows, relation_cols = np.stack(mask_info["id_same_relation"]).T
                repel_rows = np.concatenate([concept_rows, relation_rows])
                repel_cols = np.concatenate([concept_cols, relation_cols])
            else:
                repel_rows, repel_cols = concept_rows, concept_cols
            all_rows, all_cols = get_triu_ids(n_masks)
            all_tuples = [(row, col) for row, col in zip(all_rows, all_cols)]
            repel_tuples = [(row, col) for row, col in zip(repel_rows, repel_cols)]
            compat_tuples = [ele for ele in all_tuples if ele not in repel_tuples]
            if len(compat_tuples) > 0:
                compat_rows, compat_cols = np.array(compat_tuples).T
                return compat_rows, compat_cols
            else:
                return [], []

        relation_merge_mode = kwargs["relation_merge_mode"] if "relation_merge_mode" in kwargs else "threshold"

        # Perform SGLD w.r.t. mask and then w.r.t. image:
        assert len(train_input.shape) == 5
        assert len(train_target.shape) == 5
        assert len(test_input.shape) == 5
        input = torch.cat([train_input, test_input], 1)
        is_return_E = kwargs["is_return_E"] if "is_return_E" in kwargs and kwargs["is_return_E"] is True else False

        if is_return_E:
            recons, mask_dict, E_all = self.forward_NN(input, **kwargs)
        else:
            recons, mask_dict = self.forward_NN(input, **kwargs)
        device = input.device

        mask = torch.stack(list(mask_dict.values()), 2)  # [B_task, B_example, n_masks, 1, H, W]
        n_masks = len(mask_dict)
        mask_expand_0 = mask[:,:,:,None]  # [B_task, B_example, n_masks, 1, 1, H, W]
        mask_expand_1 = mask[:,:,None]  # [B_task, B_example, 1, n_masks, 1, H, W]
        distance_matrix_batch = get_soft_Jaccard_distance(mask_expand_0, mask_expand_1, dim=(-3,-2,-1))  # [B_task, B_example, n_masks, n_masks]
        distance_matrix_mean = distance_matrix_batch.mean(1)  # [B_task, n_masks, n_masks]

        mask_info = self.get_mask_info()
        compat_rows, compat_cols = get_compat_ids(mask_info)
        if len(compat_rows) > 0 and len(compat_cols) > 0:
            distance_compat = distance_matrix_mean[:, compat_rows, compat_cols]  # [B_task, n_compat]

            if relation_merge_mode == "threshold":
                relation_threshold = 0.5
                merged_mask = distance_matrix_mean < relation_threshold  # [B_task, n_masks, n_masks]
                compat_mask = torch.zeros(distance_matrix_mean.shape).bool().to(device)
                compat_mask[:, compat_rows, compat_cols] = True
                mask_combined = merged_mask & compat_mask  # [B_task, n_masks, n_masks]

                test_pred = -torch.ones(train_target[:,:1].shape).to(device)  # Default: if it is all -1, means no prediction
                for i in range(mask_combined.shape[0]):
                    combined_rows, combined_cols = to_np_array(*torch.where(mask_combined[i]), full_reduce=False)
                    unique_ids = np.unique(np.concatenate([combined_rows, combined_cols]))
                    if len(unique_ids) == 0:
                        continue
                    selected_mask = mask[i, :-1, unique_ids]
                    train_target_aug = train_target[i,:,None]
                    selected_distance = get_soft_Jaccard_distance(selected_mask, train_target_aug, dim=(-3,-2,-1))  # [n_train, n_unique_ids]
                    nearest_id = unique_ids[selected_distance.mean(0).argmin().item()]
                    test_pred[i] = mask[i, -1:, nearest_id]
            else:
                raise Exception("relation_merge_mode '{}' is not valid!".format(relation_merge_mode))
        else:
            test_pred = -torch.ones(train_target[:,:1].shape).to(device)
        if is_return_E:
            return recons, mask_dict, test_pred, E_all
        else:
            return recons, mask_dict, test_pred


    def forward_NN_gnn(
        self,
        train_input,
        train_target,
        test_input,
        **kwargs
    ):
        # Perform SGLD w.r.t. mask and then w.r.t. image:
        assert len(train_input.shape) == 5
        assert len(train_target.shape) == 5
        assert len(test_input.shape) == 5
        input = torch.cat([train_input, test_input], 1)

        # If there is an additional task-batch dimension, combine that with the example-batch:
        batch_shape = tuple(train_input.shape[:2])
        batch_shape_combined = tuple(input.shape[:2])
        train_input = train_input.reshape(-1, *train_input.shape[-3:])
        train_target = train_target.reshape(-1, *train_target.shape[-3:])
        test_input = test_input.reshape(-1, *test_input.shape[-3:])
        input = input.reshape(-1, *input.shape[-3:])
        is_return_E = kwargs["is_return_E"] if "is_return_E" in kwargs and kwargs["is_return_E"] is True else False

        # Setting up the hyperparameters:
        lambd_start = kwargs["lambd_start"] if "lambd_start" in kwargs and kwargs["lambd_start"] is not None else self.lambd_start if self.lambd_start is not None else 0.1
        lambd = kwargs["lambd"] if "lambd" in kwargs and kwargs["lambd"] is not None else self.lambd if self.lambd is not None else 0.005
        image_value_range = kwargs["image_value_range"] if "image_value_range" in kwargs and kwargs["image_value_range"] is not None else self.image_value_range if self.image_value_range is not None else "0,1"
        SGLD_is_anneal = kwargs["SGLD_is_anneal"] if "SGLD_is_anneal" in kwargs and kwargs["SGLD_is_anneal"] is not None else self.SGLD_is_anneal if self.SGLD_is_anneal is not None else True
        SGLD_is_penalize_lower = kwargs["SGLD_is_penalize_lower"] if "SGLD_is_penalize_lower" in kwargs and kwargs["SGLD_is_penalize_lower"] is not None else self.SGLD_is_penalize_lower if self.SGLD_is_penalize_lower is not None else True
        SGLD_mutual_exclusive_coef = kwargs["SGLD_mutual_exclusive_coef"] if "SGLD_mutual_exclusive_coef" in kwargs and kwargs["SGLD_mutual_exclusive_coef"] is not None else self.SGLD_mutual_exclusive_coef if self.SGLD_mutual_exclusive_coef is not None else 0
        SGLD_pixel_entropy_coef = kwargs["SGLD_pixel_entropy_coef"] if "SGLD_pixel_entropy_coef" in kwargs and kwargs["SGLD_pixel_entropy_coef"] is not None else self.SGLD_pixel_entropy_coef if self.SGLD_pixel_entropy_coef is not None else 0
        SGLD_pixel_gm_coef = kwargs["SGLD_pixel_gm_coef"] if "SGLD_pixel_gm_coef" in kwargs and kwargs["SGLD_pixel_gm_coef"] is not None else self.SGLD_pixel_gm_coef if self.SGLD_pixel_gm_coef is not None else 0
        SGLD_object_exceed_coef = kwargs["SGLD_object_exceed_coef"] if "SGLD_object_exceed_coef" in kwargs else 0
        # For selector discovery:
        SGLD_iou_batch_consistency_coef = kwargs["SGLD_iou_batch_consistency_coef"] if "SGLD_iou_batch_consistency_coef" in kwargs and kwargs["SGLD_iou_batch_consistency_coef"] is not None else self.SGLD_iou_batch_consistency_coef if self.SGLD_iou_batch_consistency_coef is not None else 0
        SGLD_iou_concept_repel_coef = kwargs["SGLD_iou_concept_repel_coef"] if "SGLD_iou_concept_repel_coef" in kwargs and kwargs["SGLD_iou_concept_repel_coef"] is not None else self.SGLD_iou_concept_repel_coef if self.SGLD_iou_concept_repel_coef is not None else 0
        SGLD_iou_relation_repel_coef = kwargs["SGLD_iou_relation_repel_coef"] if "SGLD_iou_relation_repel_coef" in kwargs and kwargs["SGLD_iou_relation_repel_coef"] is not None else self.SGLD_iou_relation_repel_coef if self.SGLD_iou_relation_repel_coef is not None else 0
        SGLD_iou_relation_overlap_coef = kwargs["SGLD_iou_relation_overlap_coef"] if "SGLD_iou_relation_overlap_coef" in kwargs and kwargs["SGLD_iou_relation_overlap_coef"] is not None else self.SGLD_iou_relation_overlap_coef if self.SGLD_iou_relation_overlap_coef is not None else 0
        SGLD_iou_attract_coef = kwargs["SGLD_iou_attract_coef"] if "SGLD_iou_attract_coef" in kwargs and kwargs["SGLD_iou_attract_coef"] is not None else self.SGLD_iou_attract_coef if self.SGLD_iou_attract_coef is not None else 0
        # Other settings:
        step_size_start = kwargs["step_size_start"] if "step_size_start" in kwargs else -1
        step_size = kwargs["step_size"] if "step_size" in kwargs and kwargs["step_size"] is not None else self.step_size if self.step_size is not None else 20
        step_size_img = kwargs["step_size_img"] if "step_size_img" in kwargs and kwargs["step_size_img"] is not None else self.step_size_img if self.step_size_img is not None else -1
        step_size_z = kwargs["step_size_z"] if "step_size_z" in kwargs and kwargs["step_size_z"] is not None else self.step_size_z if self.step_size_z is not None else 2
        step_size_zgnn = kwargs["step_size_zgnn"] if "step_size_zgnn" in kwargs and kwargs["step_size_zgnn"] is not None else self.step_size_zgnn if self.step_size_zgnn is not None else 2
        step_size_wtarget = kwargs["step_size_wtarget"] if "step_size_wtarget" in kwargs and kwargs["step_size_wtarget"] is not None else self.step_size_wtarget if self.step_size_wtarget is not None else -1
        sample_step = kwargs["sample_step"] if "sample_step" in kwargs else 60
        ensemble_size = kwargs["ensemble_size"] if "ensemble_size" in kwargs else 1
        kl_all_step = kwargs["kl_all_step"] if "kl_all_step" in kwargs else True
        is_grad = kwargs["is_grad"] if "is_grad" in kwargs else False
        w_init_type = kwargs["w_init_type"] if "w_init_type" in kwargs and kwargs["w_init_type"] is not None else self.w_init_type if self.w_init_type is not None else "random"
        indiv_sample = kwargs["indiv_sample"] if "indiv_sample" in kwargs and kwargs["indiv_sample"] is not None else self.indiv_sample if self.indiv_sample is not None else -1
        img_init_type = kwargs["img_init_type"] if "img_init_type" in kwargs else "random"
        is_return_E = kwargs["is_return_E"] if "is_return_E" in kwargs else False
        isplot = kwargs["isplot"] if "isplot" in kwargs else 0
        verbose = kwargs["verbose"] if "verbose" in kwargs else 0

        args = get_pdict()(
            lambd_start=lambd_start,
            lambd=lambd,
            image_value_range=image_value_range,
            step_size_start=step_size_start,
            step_size=step_size,
            step_size_img=step_size_img,
            step_size_z=step_size_z,
            step_size_zgnn=step_size_zgnn,
            step_size_wtarget=step_size_wtarget,
            image_size=input.shape[-2:],
            kl_all_step=kl_all_step,
            SGLD_is_anneal=SGLD_is_anneal,
            SGLD_is_penalize_lower=SGLD_is_penalize_lower,
            SGLD_object_exceed_coef=SGLD_object_exceed_coef,
            SGLD_mutual_exclusive_coef=SGLD_mutual_exclusive_coef,
            SGLD_pixel_entropy_coef=SGLD_pixel_entropy_coef,
            SGLD_pixel_gm_coef=SGLD_pixel_gm_coef,
            # Selector:
            SGLD_iou_batch_consistency_coef=SGLD_iou_batch_consistency_coef,
            SGLD_iou_concept_repel_coef=SGLD_iou_concept_repel_coef,
            SGLD_iou_relation_repel_coef=SGLD_iou_relation_repel_coef,
            SGLD_iou_relation_overlap_coef=SGLD_iou_relation_overlap_coef,
            SGLD_iou_attract_coef=SGLD_iou_attract_coef,
        )
        ebm_0 = self.ebm_dict[next(iter(self.ebm_dict))]

        with torch.enable_grad():
            ###################################################
            # 1. SGLD w.r.t. w, z and zgnn:
            ###################################################
            if self.z_mode != "None":
                args.ebm_target = "mask+z+zgnn"
            else:
                args.ebm_target = "mask+zgnn"

            if w_init_type in ["random", "input-mask", "input-gaus"] or w_init_type.startswith("k-means"):
                mask = None
            elif w_init_type == "input":
                mask_arity = len(self.nodes)
                if "obj" in ebm_0.w_type:
                    assert "mask" not in ebm_0.w_type
                    mask = tuple(train_input.detach().clone().repeat_interleave(ensemble_size, dim=0) for k in range(mask_arity))
                elif "mask" in ebm_0.w_type:
                    assert "obj" not in ebm_0.w_type
                    if train_input.shape[1] == 10:
                        mask = tuple((train_input[:,:1]!=1).detach().clone().repeat_interleave(ensemble_size, dim=0) for k in range(mask_arity))
                    elif train_input.shape[1] == 3:
                        mask = None
                    else:
                        raise
                else:
                    raise
                for k in range(mask_arity):
                    if mask is not None:
                        mask[k].requires_grad = True
            else:
                raise Exception("w_init_type '{}' is not valid!".format(w_init_type))

            (_, neg_mask_ensemble_sorted_train, z_ensemble_sorted_train, zgnn_ensemble_sorted, _), neg_out_ensemble_sorted, info = self.ground(
                train_input,
                args,
                mask=mask,
                wtarget=train_target,
                ensemble_size=ensemble_size,
                topk=1,
                w_init_type=w_init_type,
                sample_step=sample_step,
                is_grad=is_grad,
                is_return_E=is_return_E,
                batch_shape=batch_shape,
                isplot=isplot,
            )
            if verbose >= 1:
                if hasattr(self.gnn, "softmax_coef"):
                    print("softmax_coef:")
                    print(self.gnn.softmax_coef)
                print(list(self.gnn.parameters())[:4])
                print("zgnn_node:")
                if zgnn_ensemble_sorted[0] is None:
                    print(None)
                else:
                    print(zgnn_ensemble_sorted[0].squeeze())
                print("zgnn_edge:")
                print(zgnn_ensemble_sorted[1].squeeze())
                print()

            ###################################################
            # 2nd SGLD w.r.t. m', z', wtarget', given zgnn, on test examples:
            ###################################################
            args.ebm_target = "mask+z+wtarget"
            if w_init_type in ["random", "input-mask", "input-gaus"] or w_init_type.startswith("k-means"):
                mask = None
            elif w_init_type == "input":
                mask_arity = len(self.nodes)
                if "obj" in ebm_0.w_type:
                    assert "mask" not in ebm_0.w_type
                    mask = tuple(input.detach().clone().repeat_interleave(ensemble_size, dim=0) for k in range(mask_arity))
                elif "mask" in ebm_0.w_type:
                    assert "obj" not in ebm_0.w_type
                    if input.shape[1] == 10:
                        mask = tuple((input[:,:1]!=1).detach().clone().repeat_interleave(ensemble_size, dim=0) for k in range(mask_arity))
                    elif input.shape[1] == 3:
                        mask = None
                    else:
                        raise
                else:
                    raise
                for k in range(mask_arity):
                    if mask is not None:
                        mask[k].requires_grad = True
            else:
                raise Exception("w_init_type '{}' is not valid!".format(w_init_type))
            zgnn_latent = tuple(zgnn_ele[:,0] if zgnn_ele is not None else None for zgnn_ele in zgnn_ensemble_sorted)

            (_, neg_mask_ensemble_sorted, z_ensemble_sorted, _, wtarget_ensemble_sorted), neg_out_ensemble_sorted, info = self.ground(
                input,
                args,
                mask=mask,
                zgnn=zgnn_latent,
                ensemble_size=ensemble_size,
                topk=1,
                w_init_type=w_init_type,
                sample_step=sample_step,
                is_grad=is_grad,
                is_return_E=is_return_E,
                batch_shape=batch_shape_combined,
                isplot=isplot,
            )
            wtarget_pred = wtarget_ensemble_sorted[:,0]
            wtarget_pred = wtarget_pred.view(*batch_shape_combined, *wtarget_pred.shape[-3:])
            if is_return_E:
                E_all = info["E_all"]

            ###################################################
            # 3rd SGLD w.r.t. img, given mask and z:
            ###################################################
            args.ebm_target = "image"
            # Obtain the w_selected (list of w selected by the refer nodes) and a dict of all ws:
            w_op_dict = {}
            z_op_dict = {}
            w_selector = []
            nodes = list(self.nodes)
            for i, neg_mask_ele in enumerate(neg_mask_ensemble_sorted):
                neg_mask_top = neg_mask_ele[:,0]
                if self.refer_node_names is None or nodes[i] in self.refer_node_names:
                    # Accumulate the objects specified by the self.refer_node_names:
                    w_selector.append(neg_mask_top)
                w_op_dict[nodes[i]] = neg_mask_top
                if indiv_sample != -1:
                    z_op_dict[nodes[i]] = z_ensemble_sorted[i][:,0]

            # If is_reconstruct, perform a second SGLD w.r.t. img given the inferred w (and z):
            if img_init_type == "random":
                img_value_min, img_value_max = image_value_range.split(",")
                img_value_min, img_value_max = eval(img_value_min), eval(img_value_max)
                img_init = torch.rand(input.shape, device=train_input.device) * (img_value_max - img_value_min) + img_value_min
            elif img_init_type == "input":
                img_init = input.detach()
            else:
                raise

            # Specify the mask according to the refer nodes:
            neg_mask_latent = []
            mask_info = self.get_mask_info()
            for i, neg_mask_ele in enumerate(neg_mask_ensemble_sorted):
                if self.refer_node_names is None or nodes[i] in self.refer_node_names:
                    if mask_info["id_to_type"][i][0] == "relation" and self.is_relation_z is False:
                        neg_mask_latent.append(None)
                    else:
                        neg_mask_latent.append(neg_mask_ele[:,0])
                else:
                    # If not refer nodes, then turn off the mask:
                    neg_mask_latent.append(torch.zeros(neg_mask_ele[:,0].shape).to(device))

            # Specify the z:
            z_latent = tuple(ele[:,0] if ele is not None else None for ele in z_ensemble_sorted) if "z" in args.ebm_target else None

            if indiv_sample != -1:
                # Go through each EBM
                pred = None
                ebm_dict = self.get_ebms()
                for key in w_op_dict.keys():
                    model = list(ebm_dict[key].values())[0]
                    (img_ensemble_sorted, _, _, _, _), neg_out_ensemble_sorted = model.ground(
                        img_init,
                        args,
                        mask=tuple([w_op_dict[key]]),
                        z=tuple([z_op_dict[key]]),
                        ensemble_size=ensemble_size,
                        topk=1,
                        sample_step=indiv_sample,
                        is_grad=is_grad,
                        isplot=isplot,
                    )
                    if pred is None:
                        pred = img_ensemble_sorted[:, 0] * w_op_dict[key]
                    else:
                        pred = pred + img_ensemble_sorted[:, 0] * w_op_dict[key]
                img_value_min, img_value_max = args.image_value_range.split(",")
                img_value_min, img_value_max = eval(img_value_min), eval(img_value_max)
                pred = pred.clamp(min=img_value_min, max=img_value_max)
            else:
                (img_ensemble_sorted, _, _, _, _), neg_out_ensemble_sorted, info = self.ground(
                    img_init,
                    args,
                    mask=neg_mask_latent,
                    z=z_latent,
                    ensemble_size=ensemble_size,
                    topk=1,
                    sample_step=sample_step,
                    is_grad=is_grad,
                    isplot=isplot,
                )
                pred = img_ensemble_sorted[:,0]

        assert pred.shape == input.shape

        # Recover the task-batch dimension:
        pred = pred.view(*batch_shape_combined, *pred.shape[-3:])
        w_op_dict = {key: item.view(*batch_shape_combined, *item.shape[-3:]) for key, item in w_op_dict.items()}

        # Store the results in a cache, to be used by the policy:
        if is_return_E:
            return pred, w_op_dict, wtarget_pred, E_all
        else:
            return pred, w_op_dict, wtarget_pred


    # Pivot and refer nodes:
    def set_pivot_nodes(self, node_names):
        """Set pivot nodes for the concept pattern, which are used for identifying
        a pivot in the concept_graph.
        """
        if not isinstance(node_names, list):
            node_names = [node_names]
        node_name_list = []
        for node_name in node_names:
            node_name = self.get_node_name(node_name)
            if node_name in self.nodes:
                node_name_list.append(node_name)
        self.pivot_node_names = node_name_list
        self.clear_instance()
        return self


    def set_refer_nodes(self, node_names):
        """Set refer nodes for the concept pattern, which are used for identifying
        the nodes intended to refer to in the concept_graph.
        """
        if not isinstance(node_names, list):
            node_names = [node_names]
        node_name_list = []
        for node_name in node_names:
            node_name = self.get_node_name(node_name)
            if node_name in self.nodes:
                node_name_list.append(node_name)
        self.refer_node_names = node_name_list
        return self


    # Clear stuff:
    def clear_instance(self):
        """If pivot_node_names is set, clear the value of all other nodes
        (since it is graph pattern, we want to refer to all other nodes using relation w.r.t. pivot_node)."""
        if self.pivot_node_names is not None:
            for node_name in self.nodes:
                if node_name not in self.pivot_node_names:
                    self.remove_node_value(node_name)
        else:
            print("Pivot nodes are not set. Do not clear any content of the nodes.")
        return self


    def add_obj(self, concept, *opsc, **kwargs):
        """Add a concept node (representing an object) to the Concept_Pattern.

        Args:
            concept: concept to be added to the Concept_Pattern
            *opsc: a list of existing concept nodes (whose default must be DEFAULT_OBJ_TYPE) whose mode
                will be replaced by the concept's mode.
            **kwargs: kwargs for init_ebm() if the concept does not exist in the selector.
        """
        if len(opsc) == 0:
            # Add a concept node and do not merge with any existing concept node:
            if "add_obj_name" in kwargs:
                obj_name = kwargs["add_obj_name"]
            else:
                obj_name = get_next_available_key([node_name.split(":")[0] for node_name in self.nodes], "obj", is_underscore=True)
            placeholder = Placeholder(concept.name)
            if concept.name not in self.ebm_dict:
                self.init_ebm(
                    method="random",
                    mode=concept.name,
                    ebm_mode="concept",
                    ebm_model_type="CEBM",
                    **kwargs
                )
            placeholder.set_ebm_key(concept.name)
            self.add_node("{}:{}".format(obj_name, concept.name), value=placeholder, type="obj")
        else:
            # Replacing the default concept type in the nodes in opsc by the concept's type:
            for op_name in opsc:
                op_name = self.get_node_name(op_name)
                op_mode = op_name.split(":")[-1]
                assert op_mode == DEFAULT_OBJ_TYPE, "To assign concept to existing opsc, the opsc '{}' must have DEFAULT_OBJ_TYPE of '{}', not '{}'.".format(op_name, DEFAULT_OBJ_TYPE, op_mode)
                placeholder = self.get_node_content(op_name)
                if concept.name not in self.ebm_dict:
                    self.init_ebm(
                        method="random",
                        mode=concept.name,
                        ebm_mode="concept",
                        ebm_model_type="CEBM",
                        **kwargs
                    )
                placeholder.change_mode(concept.name, new_ebm_key=concept.name)
                nx.relabel_nodes(self, {op_name: "{}:{}".format(op_name.split(":")[0], concept.name)}, copy=False)
        # Important: reset the forward results cache
        if self.cache_forward:
            self.forward_cache = {}
        return self


    def remove_attr(self, attr_name):
        """Remove an attribute and all its descendant attributes."""
        super(Concept_Pattern, self).remove_attr(attr_name, change_root=False)
        if self.pivot_node_names is not None and attr_name in self.pivot_node_names:
            self.pivot_node_names.remove(attr_name)
        if self.refer_node_names is not None and attr_name in self.refer_node_names:
            self.refer_node_names.remove(attr_name)
        return self


    def remove_attrs_from(self, attr_names):
        """Remove multiple nodes from the graph, also remove them in pivot and refer nodes if they are inside."""
        if not isinstance(attr_names, list):
            attr_names = [attr_names]
        for attr_name in attr_names:
            self.remove_attr(attr_name)
        return self


    def parameters(self):
        """Obtain the parameters of the relevant ebms in the self.ebm_dict."""
        keys = []
        for src, ebm_dict in self.get_ebms().items():
            if ebm_dict is not None:
                keys += list(ebm_dict.keys())
        keys = remove_duplicates(keys)
        return itertools.chain.from_iterable([self.ebm_dict[key].parameters() for key in keys])


    def to(self, device):
        super(Concept_Pattern, self).to(device)
        self.device = device
        if hasattr(self, "ebm_dict"):
            for key in self.ebm_dict:
                self.ebm_dict[key].to(device)
            if self.ebm_dict.__class__.__name__ == "Shared_Param_Dict":
                self.ebm_dict.to(device)
        if hasattr(self, "gnn"):
            self.gnn.to(device)
        return self


    @property
    def mask_arity(self):
        return len(self.nodes)


    # Printing:
    def __str__(self):
        string = "nodes={}, edges={}".format(len(self.nodes), len(self.edges))
        if self.pivot_node_names is not None:
            string += ", pivot={}".format(self.pivot_node_names)
        if self.refer_node_names is not None:
            string += ", refer={}".format(self.refer_node_names)
        return "Concept_Pattern({})".format(string)


    def __repr__(self):
        if IS_VIEW:
            if len(self.nodes) > 0:
                self.draw()
        return self.__str__()


# ### 1.4.3 Concept_Ensemble

# In[ ]:


class Concept_Ensemble(BaseGraph):
    """An ensemble of concepts, including their inter-concept relations."""
    def __init__(self, concepts=None):
        super(Concept_Ensemble, self).__init__()
        self.add_concepts(concepts)


    def add_concept(self, concept, name):
        """Add a single concept."""
        self.add_node(name, value=concept, type="concept")
        return self


    def add_concepts(self, concepts):
        """Add multiple concepts."""
        if concepts is not None:
            for name, concept in concepts.items():
                self.add_concept(concept, name=name)
        return self


    def get_concept(self, name):
        """Obtain the concept according to its name."""
        return self.get_node_content(name)


    @property
    def concept_names(self):
        """Return all concept names in the emsemble."""
        return list(self.nodes)


    @property
    def concepts(self):
        """Return all concepts in the emsemble."""
        concepts = OrderedDict()
        for name in self.concept_names:
            concepts[name] = self.get_concept(name)
        return concepts


    @property
    def relations(self):
        """Return all relations between inter-concept objects in the emsemble."""
        relations = {}
        for edge in self.edges:
            if edge[-1] == 1:
                relations[(edge[0], edge[1])] = self.edges[edge]["value"]
        return relations


    @property
    def theories(self):
        """Return all theory strings between inter-concept objects in the emsemble."""
        theories = {}
        for edge in self.edges:
            if edge[-1] == 0:
                theories[(edge[0], edge[1])] = self.edges[edge]["value"]
        return theories


    def add_relations(self, concept1_name, concept2_name, OPERATORS, allowed_types=["Bool"]):
        """Add all possible relations between objects of concept1 and objects of concept2."""
        if not isinstance(allowed_types, list):
            allowed_types = [allowed_types]
        for obj1_name, obj1 in self.get_concept(concept1_name).objs.items():
            for obj2_name, obj2 in self.get_concept(concept2_name).objs.items():
                for name, op in OPERATORS.items():
                    output_mode = op.get_to_outnode(op.name).split(":")[-1]
                    if "Bool" in allowed_types and output_mode == "Bool":
                        is_valid = op(obj1, obj2)
                    elif "Op" in allowed_types and output_mode != "Bool" and len(op.dangling_nodes) == 0 and len(op.input_placeholder_nodes) == 1:
                        obj1_trans = op(obj1)
                        is_valid = obj1_trans == obj2
                    if is_valid:
                        edge_key = (concept1_name, concept2_name, 1)
                        if edge_key not in self.edges:
                            self.add_edge(concept1_name, concept2_name, 1, value={}, type="inter-concept-relation")
                        record_data(self.edges[edge_key]["value"], [name], [(obj1_name, obj2_name)])
        return self


    def add_theories(self, concept1_name, concept2_name, pair_list):
        """Add theories as specified by pair_list between objects in concept1 and concept2"""
        for pair in pair_list:
            edge_key = (concept1_name, concept2_name, 0)
            if edge_key not in self.edges:
                self.add_edge(concept1_name, concept2_name, 0, value={}, type="inter-concept-theory")
            input_obj_name, target_obj_name = pair["input_obj_name"], pair["target_obj_name"]
            self.edges[edge_key]["value"][(input_obj_name, target_obj_name)] = pair["op_names"]
        return self


    def __str__(self):
        string = ", ".join(["{}: {}".format(name, self.get_concept(name).name) for name in self.concept_names])
        return "Concept_Ensemble({})".format(string)


    def __repr__(self):
        self.draw()
        return self.__str__()


# In[ ]:


# Batch processing:
def add_relations(concepts, OPERATORS, allowed_types=["Bool"]):
    """Endow each concept the valid relations among objects."""
    if not isinstance(allowed_types, list):
        allowed_types = [allowed_types]
    if isinstance(concepts, list):
        return [add_relations(concept, OPERATORS, allowed_types=allowed_types) for concept in concepts]
    if not isinstance(concepts, dict):
        return concepts.add_relations(OPERATORS, allowed_types=allowed_types)
    else:
        return OrderedDict([[name, concept.add_relations(OPERATORS, allowed_types=allowed_types)] for name, concept in concepts.items()])
    
    
def get_refer_nodes(concepts, concept_pattern):
    """For each concept, obtain refer node."""
    if not isinstance(concepts, dict):
        return concepts.get_refer_nodes(concept_pattern)
    else:
        return OrderedDict([[name, concept.get_refer_nodes(concept_pattern)] for name, concept in concepts.items()])


def to_Concept(patches):
    """PyTorch tensors to Concept list."""
    return {"obj_{}:Image".format(i): CONCEPTS[DEFAULT_OBJ_TYPE].copy().set_node_value(value).set_node_value(pos, "pos") 
                  for i, (value, pos) in enumerate(patches)}


def parse_obj(concepts, is_colordiff=True, is_diag=True, verbose=False):
    """Endow concepts with parsed objects"""
    if isinstance(concepts, list):
        return [parse_obj(concept, is_colordiff=is_colordiff, is_diag=is_diag, verbose=verbose) for concept in concepts]
    if not isinstance(concepts, dict):
        return parse_obj_ele(concepts, is_colordiff=is_colordiff, is_diag=is_diag, verbose=verbose)
    else:
        return OrderedDict([[name, parse_obj_ele(concept, is_colordiff=is_colordiff, is_diag=is_diag, verbose=verbose)] for name, concept in concepts.items()])


def parse_obj_ele(concept, is_colordiff=True, is_diag=True, verbose=False):
    """Endow a concept with parsed objects"""
    if isinstance(concept, Concept):
        tensor = concept.get_node_value()
    else:
        tensor = concept
    if is_colordiff:
        patches = find_connected_components_colordiff(tensor, is_diag=is_diag)
    else:
        patches = find_connected_components(tensor, is_diag=is_diag)
    objs = to_Concept(patches)
    if len(objs) > 1:
        for obj_name, obj in objs.items():
            if verbose:
                print('Displaying {}:'.format(obj_name))
                display(obj)
                print()
            concept.add_obj(obj, obj_name=obj_name)
    elif len(objs) == 1:
        obj_name = list(objs.keys())[0]
        obj = objs[obj_name]
        if not (obj.get_node_value("pos") == concept.get_node_value("pos")).all():
            concept.add_obj(obj, obj_name=obj_name)
    return concept


def endow_objs_from_pairing_ele(input, target, pair_list):
    """Add objects to input and target (assuming originally no obj parsing)."""
    input_graph = input.copy()
    target_graph = target.copy()
    for pair in pair_list:
        input_obj = pair["input"]
        target_obj = pair["target"]
        input_obj_name = input_graph.add_obj(input_obj)
        target_obj_name = target_graph.add_obj(target_obj)
        pair["input_obj_name"] = input_obj_name
        pair["target_obj_name"] = target_obj_name
    return input_graph, target_graph


def endow_objs_from_pairing(inputs, targets, pair_list_all, is_inter_relation=False):
    """Return a concept_ensemble where the each pair of (input, target) are endowed objects and their relations."""
    concept_ensemble = Concept_Ensemble()
    if not isinstance(inputs, dict):
        inputs = {0: inputs}
    if not isinstance(targets, dict):
        targets = {0: targets}
    for key, input in inputs.items():
        target = targets[key]
        pair_list = pair_list_all[key]["pair_list"]
        input_graph, target_graph = endow_objs_from_pairing_ele(input, target, pair_list)
        concept_ensemble.add_concept(input_graph, (key, "input"))
        concept_ensemble.add_concept(target_graph, (key, "target"))
        concept_ensemble.add_theories((key, "input"), (key, "target"), pair_list)
        if is_inter_relation:
            concept_ensemble.add_relations((key, "input"), (key, "target"), OPERATORS)
    return concept_ensemble


def get_pair_PyG_data(
    input,
    target,
    OPERATORS,
    parse_pair_ele,
    allowed_attr="obj",
    repr_format="onehot",
    cache_dirname=None,
):
    """Given input and target, parse pair_list and return edge_index and edge_attr on the relation between objects in input and targets."""
    if len(input.obj_names) == 0:
        input = parse_obj_ele(input)
    if len(target.obj_names) == 0:
        target = parse_obj_ele(target)
    pair_list, info = parse_pair_ele(input, target, use_given_objs=False, cache_dirname=cache_dirname, isplot=False)

    # Build (composite) objects:
    for pair in pair_list:
        input_patch = pair["input"]
        target_patch = pair["target"]

        obj_names_input, unexplained_input = input.parse_comp_obj(input_patch)
        assert unexplained_input.sum() == 0
        if len(obj_names_input) > 1:
            obj_name_input = input.combine_objs(obj_names_input)
        else:
            obj_name_input = obj_names_input[0]
        pair["obj_name_input"] = obj_name_input
        obj_names_target, unexplained_target = target.parse_comp_obj(target_patch)
        assert unexplained_target.sum() == 0
        if len(obj_names_target) > 1:
            obj_name_target = target.combine_objs(obj_names_target)
        else:
            obj_name_target = obj_names_target[0]
        pair["obj_name_target"] = obj_name_target

    # Build PyG data:
    edge_index = []
    edge_attr = []

    input_nodes_sorted = input.get_graph(allowed_attr).topological_sort
    num_input_nodes = len(input_nodes_sorted)
    target_nodes_sorted = target.get_graph(allowed_attr).topological_sort
    
    # x denotes whether the node is input (0) or target node:
    x = torch.zeros(num_input_nodes + len(target_nodes_sorted), 2)
    x[:num_input_nodes, 0] = 1
    x[num_input_nodes:, 1] = 1

    # Build relations between objects in input and objects in target:
    for pair in pair_list:
        input_patch = pair["input"]
        target_patch = pair["target"]
        op_names_pair = pair["op_names"]
        source_id = input_nodes_sorted.index(pair["obj_name_input"])
        target_id = num_input_nodes + target_nodes_sorted.index(pair["obj_name_target"])
        edge_index.append([source_id, target_id])
        if repr_format == "onehot":
            edge_attr_vec = torch.zeros(len(OPERATORS) + 4)
            for op_name in OPERATORS:
                for op_name_pair in op_names_pair:
                    if op_name in op_name_pair:
                        op_id = list(OPERATORS.keys()).index(op_name)
                        edge_attr_vec[op_id] = 1
                    if "changeColor" in op_name_pair:
                        op_id = list(OPERATORS.keys()).index("Draw")
                        edge_attr_vec[op_id] = 1
        elif repr_format == "embedding":
            edge_attr_vec = torch.zeros(REPR_DIM)
            for op_name in OPERATORS:
                for op_name_pair in op_names_pair:
                    if op_name in op_name_pair:
                        edge_attr_vec = edge_attr_vec + OPERATORS[op_name].get_node_repr()
                        break
        else:
            raise Exception("repr_format {} is not supported!".format(repr_format))
        edge_attr.append(edge_attr_vec)
    if len(edge_index) > 0:
        edge_index = to_Variable(edge_index).long().T.to(input.device)
        edge_attr = torch.stack(edge_attr).to(input.device)
    else:
        edge_index = torch.zeros(2, 0).long().to(input.device)
        if repr_format == "onehot":
            edge_attr = torch.zeros(0, len(OPERATORS) + 4).to(input.device)
        elif repr_format == "embedding":
            edge_attr = torch.zeros(0, REPR_DIM).to(input.device)
        else:
            raise

    # Combine with relations among objects in input and objects in target:
    if len(input.get_relations()) == 0:
        input.add_relations(OPERATORS)
    input_data = input.get_PyG_data(OPERATORS, allowed_attr=allowed_attr, repr_format=repr_format)
    if len(target.get_relations()) == 0:
        target.add_relations(OPERATORS)
    target_data = target.get_PyG_data(OPERATORS, allowed_attr=allowed_attr, repr_format=repr_format)
#     x_repr = torch.cat([input_data.x, target_data.x], 0)
#     x = torch.cat([x, x_repr], -1)
    edge_index = torch.cat([edge_index, input_data.edge_index, num_input_nodes + target_data.edge_index], -1)
    edge_attr = torch.cat([edge_attr, input_data.edge_attr, target_data.edge_attr])

    return x, edge_index, edge_attr


# ## 2.1 Important methods:

# In[ ]:


def discover_relations(
    inputs,
    num_copies=100,
    max_operators=8,
    add_full_concept=True,
    add_attr_prob=0.2,
    mode="stack",
    input_mode_dict=None,
    OPERATORS=None,
    CONCEPTS=None,
    ):
    """Find a list of Graph() that returns True in one of its results, for all examples in positives_inputs."""
    # Find relations:
    relation = Graph()
    ## Add initial concepts:
    key = list(inputs[0].keys())[0]
    for input_arg in inputs:
        relation.add_subgraph(input_arg[key], add_full_concept=add_full_concept)

    graph_dict = OrderedDict([[i, deepcopy(relation)] for i in range(num_copies)])
    concepts = combine_dicts([OPERATORS])
    graph_id_true = {}
    is_stop = False
    for k in range(max_operators):
        for i in range(num_copies):
            if i not in graph_id_true:
                graph_dict[i] = add_operator_random(graph_dict[i], concepts, add_attr_prob=add_attr_prob, input_mode_dict=input_mode_dict, OPERATORS=OPERATORS, CONCEPTS=CONCEPTS)
                results = graph_dict[i](*inputs, is_output_all=True)
                is_result_true, node_true = check_result_true(results)
                if is_result_true:
                    # Preserve minimal subgraph that contains node_true:
                    G = graph_dict[i].copy().preserve_subgraph(node_true, level="node")
                    if len(G.input_placeholder_nodes) == len(inputs):
                        graph_id_true[i] = k
                        graph_dict[i] = G
                        if mode == "exists":
                            is_stop = True
                            break
                        elif mode == "stack":
                            pass
                        else:
                            raise
                    else:
                        graph_dict[i].preserve_subgraph(node_true, level="operator")
        if is_stop:
            break
    graph_list = [graph for i, graph in graph_dict.items() if i in graph_id_true]
    return graph_list


def compose_relations(graph_list, max_operators=2):
    """Unify multiple graphs into a single relational operator graph."""
    relation = Graph()
    for graph in graph_list:
        if len(graph.operators) <= max_operators:
            # Remove duplicate operators (temporary):
            operator_to_remove = []
            for operator in graph.operators:
                if operator in relation.operators:
                    operator_to_remove.append(operator)
            if len(operator_to_remove) > 0:
                graph = graph.copy().remove_subgraph(operator_to_remove, is_rename=False)
            # Compose:
            relation.compose(graph)
    bool_output_node = relation.add_And_over_bool()
    length = len(relation.nodes)
    if bool_output_node is not None:
        relation.preserve_subgraph(bool_output_node, level="node")
    else:
        raise Exception("The relation does not have a Boolean output node!")
    relation.remove_subgraph(relation.operators_dangling)
    if len(relation.nodes) != length:
        print("The composed relation had {} dangling outputs that are removed.".format(length - len(relation.nodes)))
    return relation


def create_new_concept(
    positive_inputs,
    relation,
    name=None,
    inherit_from=[DEFAULT_OBJ_TYPE],
    is_cuda=False,
):
    """Create a new concept based on the data and discovered relational operator graph."""
    # Add basic properties:
    if name is None:
        name = get_next_available_key(NEW_CONCEPTS, "CNew",
                                      suffix="Id", is_underscore=False)
    kwargs = {}
    kwargs["name"] = name
    kwargs["repr"] = to_Variable(torch.rand(REPR_DIM), is_cuda=is_cuda)
    kwargs["inherit_from"] = inherit_from
    kwargs["value"] = Placeholder(Tensor(dtype="cat"))
    kwargs["attr"] = OrderedDict()

    # Add attributes based on positives_parse_dict:
    key = next(iter(positive_inputs[0]))
    for input_arg in positive_inputs:
        concept = input_arg[key]
        concept_name = concept.name
        concept_name_lower = concept_name[0].lower() + concept_name[1:]
        current_keys = Counter([split_string(string)[0] for string in list(kwargs["attr"].keys())])
        if concept_name_lower not in current_keys:
            current_keys[concept_name_lower] = 1
            suffix = ""
        else:
            suffix = current_keys[concept_name_lower]
            current_keys[concept_name_lower] += 1
        kwargs["attr"]["{}{}".format(concept_name_lower, suffix)] = Placeholder(concept_name)
    # Add relation:
    kwargs["re"] = {tuple(kwargs["attr"].keys()): relation}
    # Add position attribute:
    kwargs["attr"]["pos"] = Placeholder("Pos")
    return Concept(**kwargs)


def add_operator_random(
    graph,
    concepts,
    dangling_mode={"possible": 0.2, "dangling": 0.8},
    add_attr_prob=0.2,
    input_mode_dict=None,
    OPERATORS=None,
    CONCEPTS=None,
):
    """Return a new graph randomly adding a valid operator or expand attributes of an out_node, according to the 
    probabilities assigned.
    """
    output_nodes_dict = graph.get_output_nodes(["input", "attr", "fun-out"], dangling_mode=dangling_mode)
    output_nodes, priority_score = list(output_nodes_dict.keys()), np.array(list(output_nodes_dict.values()))
    priority_score = priority_score / priority_score.sum()

    for i in range(500):
        p0 = np.random.rand()
        ## Expand attributes from current out_nodes:
        if p0 < add_attr_prob:
            output_node_chosen = np.random.choice(output_nodes)
            output_node_mode = output_node_chosen.split(":")[-1]
            attrs = concepts[output_node_mode].child_nodes(output_node_mode)
            if len(attrs) == 0:
                continue
            attr_chosen = np.random.choice(attrs)
            G = deepcopy(graph)
            G.add_get_attr(output_node_chosen, attr_chosen)

        ## Add operators on current out_nodes
        else:
            p = np.random.rand()
            if p < 0.2:
                arity = 1
                nodes = np.random.choice(output_nodes, size=1, replace=False, p=priority_score).tolist()
            elif p > 0.8:
                arity = 3
                if len(output_nodes) < arity:
                    continue
                nodes = np.random.choice(output_nodes, size=3, replace=False, p=priority_score).tolist()
            else:
                arity = 2
                nodes = np.random.choice(output_nodes, size=2, replace=False, p=priority_score).tolist()
            options = find_valid_operators(nodes, operators=OPERATORS, concepts=CONCEPTS, input_mode_dict=input_mode_dict, arity=arity,
                                           exclude=["Identity", "DrawOn", "Draw", "Move"])
            if len(options) == 0:
                continue
            key = np.random.choice(list(options.keys()))
            G = deepcopy(graph)
            G.add_subgraph(OPERATORS[key])
            for source, target in options[key]:
                if G.rename_mapping is not None:
                    for operator_name, new_operator_name in G.rename_mapping.items():
                        if target.startswith(operator_name):
                            target = new_operator_name + target[len(operator_name):]
                            break
                G.connect_nodes(source, target)
        break

    if "G" not in locals():
        return graph
    else:
        return G


# ## 3. Test:

# In[ ]:


if __name__ == "__main__":
    # Testing selector augmented with EBMs:
    IS_CUDA = True
    num_colors = 10
    CONCEPTS["Image"] = Concept(name="Image",
        repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
        inherit_to=["c0", "c1"],
        value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    )

    CONCEPTS["c0"] = Concept(name="c0",
        repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
        inherit_from=["Image"],
        value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    )

    CONCEPTS["c1"] = Concept(name="c1",
        repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
        inherit_from=["Image"],
        value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    )

    OPERATORS["r0"] = Graph(name="r0",
        repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
        forward={"args": [Placeholder("Image"), Placeholder("Image")],
                 "output": Placeholder("Bool"),
                 "fun": lambda x: x,
                })

    OPERATORS["r1"] = Graph(name="r1",
        repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
        forward={"args": [Placeholder("Image"), Placeholder("Image")],
                 "output": Placeholder("Bool"),
                 "fun": lambda x: x,
                })

    c_pattern = Concept_Pattern(
        name=None,
        value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
        attr={
            "obj_0": Placeholder("c0"),
            "obj_1": Placeholder("c1"),
            "obj_2": Placeholder("c1"),
        },
        re={
            ("obj_0", "obj_1"): "r0",
            ("obj_1", "obj_2"): "r1",
        },
        pivot_node_names = ["obj_0"],
    )
    self = c_pattern
    self.init_ebms()
    ebm_dict_all = self.get_ebms()

    from reasoning.concept_env.BabyARC.code.dataset.dataset import *
    from reasoning.util import get_root_dir
    from reasoning.util import to_Variable_recur, visualize_dataset, visualize_matrices

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # global vars
    RUN_AS_CREATOR = False
    ARC_OBJ_LOADED = False
    DEMO_MAX_ARC_OBJS = 500
    pp = pprint.PrettyPrinter(indent=4)
    import logging
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                        datefmt="%Y-%m-%d %H:%M")
    logger = logging.getLogger(__name__)

    try:
        arc_obj_dir = os.path.join(get_root_dir(), 'concept_env/datasets/arc_objs.pt')
        arc_objs = torch.load(arc_obj_dir)
        logger.info("SUCCESS! You loaded the pre-collected object file from ARC!")
        ARC_OBJ_LOADED = True
    except:
        logger.info("Please check if obejct file in the directory indicated above!")
        logger.info(f"WARNING: Please get those pre-collected ARC objects in {arc_obj_dir}!")
        logger.info("You can download this file from: https://drive.google.com/file/d/1dZhT1cUFGvivJbSTwnqjou2uilLXffGY/view?usp=sharing")

    # Same Color + IsTouch + Move
    dataset_engine =         BabyARCDataset(pretrained_obj_cache=os.path.join(get_root_dir(), 'concept_env/datasets/arc_objs.pt'),
                       save_directory="./BabyARCDataset/",
                       object_limit=1, noise_level=0, canvas_size=8)

    def generate_babyarc_selector_task(
        dataset_engine,
        selector_dict,
        concept_collection=["line", "Lshape", "rectangle", "rectangleSolid"],
        is_plot=False,
    ):

        canvas_dict = dataset_engine.sample_single_canvas_by_core_edges(
            selector_dict,
            allow_connect=True, is_plot=False, rainbow_prob=0.0,
            concept_collection=concept_collection,
        )
        if canvas_dict == -1:
            return -1
        else:
            if is_plot:
                canvas = Canvas(
                    repre_dict=canvas_dict
                )
                canvas.render()

        return_dict = OrderedDict({
            "obj_masks" : {},
            "obj_relations" : {},
        })
        for k, v in canvas_dict["node_id_map"].items():
            return_dict["obj_masks"][k] = canvas_dict["id_object_mask"][v]
        return_dict["obj_relations"] = canvas_dict["partial_relation_edges"]
        return_dict["image_t"] = canvas_dict["image_t"]
        return return_dict

    input = torch.rand(128,10,8,8)
    w = [torch.rand(128,1,8,8) for _ in range(3)]
    energy = self.ebm_forward(input, w)
    print(self.info.keys())

