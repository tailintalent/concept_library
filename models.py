#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
from numbers import Number
import numpy as np
import pdb
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dists
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from typing import Union

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from concept_library.util import filter_kwargs, gather_broadcast, COLOR_LIST, Zip, init_args, to_np_array, get_filename_short, to_cpu
from concept_library.util import extend_dims, record_data, transform_dict, to_cpu, to_device_recur, init_args, get_soft_Jaccard_distance, ddeepcopy as deepcopy
from concept_library.util import get_activation, get_normalization, repeat_n, visualize_matrices, Shared_Param_Dict, MLP
from concept_library.settings import REPR_DIM


# # 1. Energy-based models:

# In[ ]:


# Helper functions
def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def id_to_tensor(ids, CONCEPTS, OPERATORS, requires_grad=False):
    """Concept string to repr tensor."""
    if not (isinstance(ids, list) or isinstance(ids, tuple)):
        ids = [ids]
    tensor = []
    for id in ids:
        if id in CONCEPTS:
            tensor.append(CONCEPTS[id].get_node_repr())
        elif id in OPERATORS:
            tensor.append(OPERATORS[id].get_node_repr())
        else:
            raise
    if requires_grad:
        return torch.stack(tensor)
    else:
        return torch.stack(tensor).detach()


# ## 1.1 Main models:

# ### 1.1.0 Load and get model:

# In[ ]:


def get_model_energy(args, device="cpu"):
    """Get EBM model according to args."""
    if args.model_type == "IGEBM":
        model = IGEBM(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            channel_base=args.channel_base,
            is_spec_norm=args.is_spec_norm,
            aggr_mode=args.aggr_mode,
        ).to(device)
    elif args.model_type == "CEBM":
        model = ConceptEBM(
            mode="operator" if args.is_two_branch else "concept",
            in_channels=args.in_channels,
            repr_dim=REPR_DIM,
            w_type=args.w_type,
            mask_mode=args.mask_mode,
            channel_base=args.channel_base,
            two_branch_mode=args.two_branch_mode,
            is_spec_norm=args.is_spec_norm,
            is_res=args.is_res,
            c_repr_mode=args.c_repr_mode,
            c_repr_first=args.c_repr_first,
            z_mode=args.z_mode,
            z_first=args.z_first,
            z_dim=args.z_dim,
            pos_embed_mode=args.pos_embed_mode,
            aggr_mode=args.aggr_mode,
            act_name=args.act_name,
            normalization_type=args.normalization_type,
            dropout=args.dropout,
            self_attn_mode=args.self_attn_mode,
            last_act_name=args.last_act_name,
            n_avg_pool=args.n_avg_pool,
        ).to(device)
    elif args.model_type == "CEBMLarge":
        model = ConceptEBMLarge(
            mode="operator" if args.is_two_branch else "concept",
            in_channels=args.in_channels,
            repr_dim=REPR_DIM,
            w_type=args.w_type,
            mask_mode=args.mask_mode,
            channel_base=args.channel_base,
            two_branch_mode=args.two_branch_mode,
            is_spec_norm=args.is_spec_norm,
            is_res=args.is_res,
            c_repr_mode=args.c_repr_mode,
            c_repr_first=args.c_repr_first,
            z_mode=args.z_mode,
            z_first=args.z_first,
            z_dim=args.z_dim,
            pos_embed_mode=args.pos_embed_mode,
            aggr_mode=args.aggr_mode,
            act_name=args.act_name,
            normalization_type=args.normalization_type,
            dropout=args.dropout,
            self_attn_mode=args.self_attn_mode,
            last_act_name=args.last_act_name,
            is_multiscale=True,
        ).to(device)
    else:
        raise Exception("model_type '{}' is not valid!".format(args.model_type))
    return model


def load_model_energy(model_dict, device="cpu"):
    """Load EBM model."""
    if model_dict is None:
        return model_dict
    model_type = model_dict["type"]
    if model_type == "IGEBM":
        model = IGEBM(
            in_channels=model_dict["in_channels"],
            n_classes=model_dict["n_classes"],
            channel_base=model_dict["channel_base"] if "channel_base" in model_dict else 128,
            is_spec_norm=model_dict["is_spec_norm"] if "is_spec_norm" in model_dict else True,
            aggr_mode=model_dict["aggr_mode"] if "aggr_mode" in model_dict else "sum",
        )
    elif model_type == "ConceptEBM":
        model = ConceptEBM(
            mode=model_dict["mode"] if "mode" in model_dict else "concept",
            in_channels=model_dict["in_channels"],
            repr_dim=model_dict["repr_dim"],
            w_type=model_dict["w_type"] if "w_type" in model_dict else "image+mask",
            mask_mode=model_dict["mask_mode"],
            channel_base=model_dict["channel_base"] if "channel_base" in model_dict else 128,
            two_branch_mode=model_dict["two_branch_mode"] if "two_branch_mode" in model_dict else "concat",
            is_spec_norm=model_dict["is_spec_norm"] if "is_spec_norm" in model_dict else "True",
            is_res=model_dict["is_res"] if "is_res" in model_dict else True,
            c_repr_mode=model_dict["c_repr_mode"] if "c_repr_mode" in model_dict else "l1",
            c_repr_first=model_dict["c_repr_first"] if "c_repr_first" in model_dict else 0,
            c_repr_base=model_dict["c_repr_base"] if "c_repr_base" in model_dict else 2,
            z_mode=model_dict["z_mode"] if "z_mode" in model_dict else "None",
            z_first=model_dict["z_first"] if "z_first" in model_dict else 2,
            z_dim=model_dict["z_dim"] if "z_dim" in model_dict else 4,
            pos_embed_mode=model_dict["pos_embed_mode"] if "pos_embed_mode" in model_dict else "None",
            aggr_mode=model_dict["aggr_mode"] if "aggr_mode" in model_dict else "sum",
            act_name=model_dict["act_name"] if "act_name" in model_dict else "leakyrelu0.2",
            normalization_type=model_dict["normalization_type"] if "normalization_type" in model_dict else "None",
            dropout=model_dict["dropout"] if "dropout" in model_dict else 0,
            self_attn_mode=model_dict["self_attn_mode"] if "self_attn_mode" in model_dict else "None",
            last_act_name=model_dict["last_act_name"] if "last_act_name" in model_dict else "None",
            n_avg_pool=model_dict["n_avg_pool"] if "n_avg_pool" in model_dict else 0,
        )
        if "c_repr" in model_dict:
            model.set_c(model_dict["c_repr"], model_dict["c_str"])
    elif model_type == "ConceptEBMLarge":
        model = ConceptEBMLarge(
            mode=model_dict["mode"] if "mode" in model_dict else "concept",
            in_channels=model_dict["in_channels"],
            repr_dim=model_dict["repr_dim"],
            w_type=model_dict["w_type"] if "w_type" in model_dict else "image+mask",
            mask_mode=model_dict["mask_mode"],
            channel_base=model_dict["channel_base"] if "channel_base" in model_dict else 128,
            two_branch_mode=model_dict["two_branch_mode"] if "two_branch_mode" in model_dict else "concat",
            is_spec_norm=model_dict["is_spec_norm"] if "is_spec_norm" in model_dict else "True",
            is_res=model_dict["is_res"] if "is_res" in model_dict else False,
            c_repr_mode=model_dict["c_repr_mode"] if "c_repr_mode" in model_dict else "l1",
            c_repr_first=model_dict["c_repr_first"] if "c_repr_first" in model_dict else 0,
            c_repr_base=model_dict["c_repr_base"] if "c_repr_base" in model_dict else 2,
            z_mode=model_dict["z_mode"] if "z_mode" in model_dict else "None",
            z_first=model_dict["z_first"] if "z_first" in model_dict else 2,
            z_dim=model_dict["z_dim"] if "z_dim" in model_dict else 4,
            pos_embed_mode=model_dict["pos_embed_mode"] if "pos_embed_mode" in model_dict else "None",
            aggr_mode=model_dict["aggr_mode"] if "aggr_mode" in model_dict else "sum",
            act_name=model_dict["act_name"] if "act_name" in model_dict else "leakyrelu0.2",
            normalization_type=model_dict["normalization_type"] if "normalization_type" in model_dict else "None",
            dropout=model_dict["dropout"] if "dropout" in model_dict else 0,
            self_attn_mode=model_dict["self_attn_mode"] if "self_attn_mode" in model_dict else "None",
            last_act_name=model_dict["last_act_name"] if "last_act_name" in model_dict else "None",
            is_multiscale=model_dict["is_multiscale"],
        )
        if "c_repr" in model_dict:
            model.set_c(model_dict["c_repr"], model_dict["c_str"])
    elif model_type == "SumEBM":
        model = SumEBM(*[load_model_energy(model_dict_ele) for model_dict_ele in model_dict["models"]])
    elif model_type == "GraphEBM":
        model = GraphEBM(
            models={key: load_model_energy(model_dict_ele) for key, model_dict_ele in model_dict["models"].items()},
            assign_dict=model_dict["assign_dict"],
            mask_arity=model_dict["mask_arity"],
        )
    elif model_type == "GNN_energy":
        model = GNN_energy(
            mode=model_dict["mode"],
            is_zgnn_node=model_dict["is_zgnn_node"],
            edge_attr_size=model_dict["edge_attr_size"],
            aggr_mode=model_dict["aggr_mode"],
            n_GN_layers=model_dict["n_GN_layers"],
            n_neurons=model_dict["n_neurons"],
            GNN_output_size=model_dict["GNN_output_size"],
            mlp_n_layers=model_dict["mlp_n_layers"],
            gnn_normalization_type=model_dict["gnn_normalization_type"],
            activation=model_dict["activation"],
            recurrent=model_dict["recurrent"],
            cnn_output_size=model_dict["cnn_output_size"],
            cnn_is_spec_norm=model_dict["cnn_is_spec_norm"],
            cnn_normalization_type=model_dict["cnn_normalization_type"],
            cnn_channel_base=model_dict["cnn_channel_base"],
            cnn_aggr_mode=model_dict["cnn_aggr_mode"],
            c_repr_dim=model_dict["c_repr_dim"],
            z_dim=model_dict["z_dim"],
            zgnn_dim=model_dict["zgnn_dim"],
            distance_loss_type=model_dict["distance_loss_type"],
            pooling_type=model_dict["pooling_type"],
            pooling_dim=model_dict["pooling_dim"],
            is_x=model_dict["is_x"] if "is_x" in model_dict else False,
        )
    else:
        raise Exception("model_type '{}' is not valid!".format(model_type))
    if model_type not in ["SumEBM", "GraphEBM"]:
        model.load_state_dict(model_dict["state_dict"])
    model.to(device)
    return model


def to_ebm_models(ebm_dict, device="cpu"):
    """Transform every element into model from model_dict."""
    if len(ebm_dict) > 0:
        if ebm_dict.__class__.__name__ == "Shared_Param_Dict":
            return ebm_dict
        elif "type" in ebm_dict and ebm_dict["type"] == "Shared_Param_Dict":
            ebm_dict = Shared_Param_Dict(
                concept_model=load_model_energy(ebm_dict["concept_model_dict"], device=device),
                relation_model=load_model_energy(ebm_dict["relation_model_dict"], device=device),
                concept_repr_dict=ebm_dict["concept_repr_dict"],
                relation_repr_dict=ebm_dict["relation_repr_dict"],
            )
        else:
            first_item = ebm_dict[next(iter(ebm_dict))]
            if isinstance(first_item, dict):
                # Tranform from model_dict to actual model:
                ebm_dict = {key: load_model_energy(model_dict, device=device) 
                            for key, model_dict in ebm_dict.items()}
    return ebm_dict


def load_best_model(
    data_record,
    keys=["mask|c_repr", "mask|c", "c_repr|mask", "c_repr|c"],
    return_id=False,
    load_epoch="best",
):
    """Load best model according to the given acc keys."""
    args = init_args(update_default_hyperparam(data_record["args"]))
    acc_mean = np.array([data_record["acc"]["acc:{}:val".format(key) if "acc:{}:val".format(key) in data_record["acc"] else "iou:{}:val".format(key)] for key in keys]).mean(0)
    acc_dict = {epoch: acc for epoch, acc in zip(data_record["acc"]["epoch:val"], acc_mean) if epoch % args.save_interval == 0}
    if load_epoch == "best":
        acc_dict_argmax = np.argmax(list(acc_dict.values()))
        acc_argmax_epoch = list(acc_dict.keys())[acc_dict_argmax]
        best_model_id = data_record["save_epoch"].index(acc_argmax_epoch)
        best_model = load_model_energy(data_record["model_dict"][best_model_id])
        acc_chosen = np.max(list(acc_dict.values()))
        assert acc_chosen == acc_dict[acc_argmax_epoch]
        print("Loaded best model for {} at epoch {}. Best mean acc: {:.6f}".format(args.dataset, acc_argmax_epoch, acc_chosen))
    elif load_epoch == "last":
        acc_argmax_epoch = data_record["save_epoch"][-1]
        best_model = load_model_energy(data_record["model_dict"][-1])
        acc_chosen = acc_dict[acc_argmax_epoch]
        print("Loaded model for {} at last epoch {} with mean acc: {:.6f}. Best mean acc: {:.6f}".format(args.dataset, acc_argmax_epoch, acc_chosen, np.max(list(acc_dict.values()))))
    else:
        acc_argmax_epoch = int(load_epoch)
        best_model_id = data_record["save_epoch"].index(acc_argmax_epoch)
        best_model = load_model_energy(data_record["model_dict"][best_model_id])
        acc_chosen = acc_dict[acc_argmax_epoch]
        print("Loaded best model for {} at epoch {} with mean acc: {:.6f}. Best mean acc: {:.6f}".format(args.dataset, acc_argmax_epoch, acc_chosen, np.max(list(acc_dict.values()))))
    if return_id:
        return best_model, best_model_id
    else:
        return best_model


def load_best_model_from_file(dirname, filename, keys=["mask|c_repr", "mask|c", "c_repr|mask", "c_repr|c"], device="cpu"):
    """Load best model according to the given acc keys from file."""
    data_record = pickle.load(open(dirname + filename, "rb"))
    model = load_best_model(data_record, keys=keys).to(device)
    model.set_repr_dict(filter_kwargs(data_record["concept_embeddings"][-1], data_record["args"]["concept_collection"]))
    return model


def load_model_atom(model_atom_str, model_type="CEBM", device="cpu"):
    """Load a dictionary of saved best models for atomic concepts and relations."""
    models = {}
    for model_str in model_atom_str.split("^"):
        model_set = set(model_str.split("+"))
        model_mode = "Load"
        if model_type == "CEBM":
            if model_set.issubset({"Line", "Lshape", "Rect"}):
                dirname = "/dfs/user/tailin/.results/ebm_5-9.1/"
                filename = "c-Line+Lshape+Rect_cz_8_model_CEBM_alpha_1_las_0.1_size_20.0_sams_60_e_r-rmb_et_mask_pl_False_nco_0.2_mask_concat_tbm_concat_cm_c2_cf_2_p_0.2_id_correct-detach_Hash_k2MkhWud_turing4.p"
            elif model_set.issubset({"Vertical", "Parallel"}):
                dirname = "/dfs/user/tailin/.results/ebm_5-9.1/"
                filename = "c-Parallel+Vertical_cz_8_model_CEBM_alpha_1_las_0.1_size_20.0_sams_60_e_r-rmb_et_mask_pl_False_nco_0.2_mask_concat_tbm_concat_cm_c2_cf_2_p_0.2_id_1_Hash_tdH7khnR_turing2.p"
            elif model_set.issubset({"SameAll", "SameShape", "SameColor", "SameRow", "SameCol", "IsInside", "IsTouch"}):
                dirname = "/dfs/user/tailin/.results/ebm_5-9.1/"
                filename = "cz_8_model_CEBM_alpha_1_las_0.1_size_20.0_sams_60_e_r-rmb_et_mask_pl_False_nco_0.2_mask_concat_tbm_concat_cm_c2_cf_2_p_0.2_id_correct-detach_Hash_tKR8Qvme_turing4.p"
            elif model_set.issubset({"Image"}):
                dirname = "/dfs/user/tailin/.results/ebm_5-9.1/"
                filename = "c-Image_cz_8_model_CEBM_alpha_1_las_0.1_size_20.0_sams_60_e_r-rmb_et_mask_pl_False_nco_0.2_mask_concat_tbm_concat_cm_c2_cf_2_p_0.2_id_correct-detach_Hash_TkXJwZhk_turing4.p"
            elif model_set.issubset({"RotateA", "RotateB", "RotateC"}):
                dirname = "/dfs/user/tailin/.results/ebm_5-9.1/"
                filename = "c-RotateA+RotateB+RotateC(Lshape)_cz_8_model_CEBM_alpha_1_las_0.1_size_20.0_sams_60_e_r-rmb_et_mask_pl_False_nco_0.2_mask_concat_tbm_concat_cm_c2_cf_2_p_0.2_id_1_Hash_f9Njs8Za_turing4.p"
            elif model_set.issubset({"AdaptRe"}):
                model_mode = "AdaptRe"
            else:
                raise Exception("No saved model for {}.".format(model_set))
        else:
            raise Exception("model_type {} is not valid!".format(model_type))
        if model_mode == "Load":
            models[model_str] = load_best_model_from_file(dirname, filename, device=device)
        elif model_mode == "AdaptRe":
            """Adaptive relations."""
            args_copy = deepcopy(args)
            args_copy.is_two_branch = True
            args_copy.model_type = "CEBM"
            models[model_str] = get_model_energy(args_copy)
        else:
            raise
    return models


def update_default_hyperparam(Dict):
    """Default hyperparameters for previous experiments, after adding these new options."""
    default_param = {
        "is_two_branch": False,
        "two_branch_mode": "concat",
        "rainbow_prob": 0,
        "max_n_distractors": 0,
        "min_n_distractors": 0,
        "allow_connect": True,
        "n_operators" : 1,
        "color_avail" : "-1",
        "transforms": "None",
        "transforms_pos": "None",
        # Training:
        "ebm_target_mode": "None",
        "ebm_target": "mask",
        "emp_target_mode": "all",
        "is_pos_repr_learnable": False,
        "p_buffer": 0.95,
        "lambd_start": -1,
        "lambd": 0.005,
        "neg_mode": "None",
        "neg_mode_coef": 0.,
        "early_stopping_patience": -1,
        "step_size_start": -1,
        "step_size_img": -1,
        "step_size_repr": -1,
        "step_size_z": 2,
        "step_size_zgnn": 2,
        "step_size_wtarget": -1,
        "is_spec_norm": "True",
        "is_res": True,
        "c_repr_mode": "l1",
        "c_repr_first": 0,
        "c_repr_base": 2,
        "aggr_mode": "sum",
        "act_name": "leakyrelu0.2",
        "normalization_type": "None",
        "dropout": 0,
        "self_attn_mode": "None",
        "last_act_name": "None",
        "n_avg_pool": 0,
        "kl_all_step": False,
        "kl_coef": 0.,
        "entropy_coef_img": 0.,
        "entropy_coef_mask": 0.,
        "entropy_coef_repr": 0.,
        "epsilon_ent": 1e-5,
        "pos_consistency_coef": 0.,
        "neg_consistency_coef": 0.,
        "emp_consistency_coef": 0.,
        # SGLD:
        "SGLD_is_anneal": False,
        "SGLD_anneal_power": 2.0,
        "SGLD_is_penalize_lower": "True",
        "SGLD_mutual_exclusive_coef": 0,
        "SGLD_fine_mutual_exclusive_coef": 0,
        "SGLD_object_exceed_coef": 0,
        "SGLD_pixel_entropy_coef": 0,
        "SGLD_mask_entropy_coef": 0,
        "SGLD_pixel_gm_coef": 0,
        # selector discovery:
        "SGLD_iou_batch_consistency_coef": 0,
        "SGLD_iou_concept_repel_coef": 0,
        "SGLD_iou_relation_repel_coef": 0,
        "SGLD_iou_relation_overlap_coef": 0,
        "SGLD_iou_attract_coef": 0,
        # Other settings:
        "w_type": "image+mask",
        "train_mode": "cd",
        "energy_mode": "standard",
        "supervised_loss_type": "mse",
        "target_loss_type": "mse",
        "cumu_mode": "harmonic",
        "channel_coef": 1,
        "empty_coef": 0.11,
        "obj_coef": 0,
        "mutual_exclusive_coef": 0,
        "pixel_entropy_coef": 0,
        "pixel_gm_coef": 0,
        "iou_batch_consistency_coef": 0,
        "iou_concept_repel_coef": 0,
        "iou_relation_repel_coef": 0,
        "iou_relation_overlap_coef": 0,
        "iou_attract_coef": 0,
        "iou_target_matching_coef": 0,
        "z_mode": "None",
        "z_first": 2,
        "z_dim": 4,
        "pos_embed_mode": "None",
        "image_value_range": "0,1",
        "w_init_type": "random",
        "indiv_sample": -1,
        "n_tasks": 128,
        "is_concat_minibatch": False,
        "to_RGB": False,
        "rescaled_size": "None",
        "rescale_mode": "nearest",
        "upsample": -1,
        "relation_merge_mode": "None",
        "is_relation_z": True,
        "connected_coef": 0,
        "connected_num_samples": 2,
        # Specific for EBM + GNN:
        "is_selector_gnn": False,
        "is_zgnn_node": False,
        "is_cross_validation": False,
        "load_pretrained_concepts": "None",
        "n_GN_layers": 2,
        "gnn_normalization_type": "None",
        "gnn_pooling_dim": 16,
        "edge_attr_size": 8,
        "cnn_output_size": 32,
        "cnn_is_spec_norm": "True",
        "train_coef": 1,
        "test_coef": 1,
        "lr_pretrained_concepts": 0,
        "parallel_mode": "None",
        "is_rewrite": False,
    }
    for key, item in default_param.items():
        if key not in Dict:
            Dict[key] = item
    return Dict


# ### 1.1.1 IGEBM:

# In[ ]:


class IGEBM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=None,
        channel_base=128,
        is_spec_norm=True,
        aggr_mode="sum",
    ):
        super(IGEBM, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.channel_base = channel_base
        self.aggr_mode = aggr_mode
        self.is_spec_norm = is_spec_norm
        if is_spec_norm:
            self.conv1 = spectral_norm(nn.Conv2d(in_channels, channel_base, 3, padding=1), std=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, channel_base, 3, padding=1)

        self.blocks = nn.ModuleList(
            [
                ResBlock(channel_base, channel_base, n_classes, downsample=True, is_spec_norm=is_spec_norm),
                ResBlock(channel_base, channel_base, n_classes, is_spec_norm=is_spec_norm),
                ResBlock(channel_base, channel_base*2, n_classes, downsample=True, is_spec_norm=is_spec_norm),
                ResBlock(channel_base*2, channel_base*2, n_classes, is_spec_norm=is_spec_norm),
                ResBlock(channel_base*2, channel_base*2, n_classes, downsample=True, is_spec_norm=is_spec_norm),
                ResBlock(channel_base*2, channel_base*2, n_classes, is_spec_norm=is_spec_norm),
            ]
        )

        self.linear = nn.Linear(channel_base*2, 1)

    def forward(self, input, class_id=None):
        out = self.conv1(input)

        out = F.leaky_relu(out, negative_slope=0.2)

        for block in self.blocks:
            out = block(out, class_id)

        out = F.relu(out)
        if self.aggr_mode == "sum":
            out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        elif self.aggr_mode == "max":
            out = out.view(out.shape[0], out.shape[1], -1).max(2)[0]
        elif self.aggr_mode == "mean":
            out = out.view(out.shape[0], out.shape[1], -1).mean(2)
        else:
            raise
        out = self.linear(out)

        return out

    @property
    def model_dict(self):
        model_dict = {"type": "IGEBM"}
        model_dict["in_channels"] = self.in_channels
        model_dict["n_classes"] = self.n_classes
        model_dict["channel_base"] = self.channel_base
        model_dict["is_spec_norm"] = self.is_spec_norm
        model_dict["aggr_mode"] = self.aggr_mode
        model_dict["state_dict"] = to_cpu(self.state_dict())
        return model_dict


# ### 1.1.2 ConceptEBM:

# In[ ]:


class ConceptEBM(nn.Module):
    """
    An EBM designed specifically to find Concepts, e.g. Rect, Line, RectSolid or
        Relations, e.g. SameShape, SameColor.
    """
    
    def __init__(
        self,
        mode="concept",
        in_channels=10,
        repr_dim=4,
        w_type="image+mask",
        mask_mode="concat",
        channel_base=128,
        two_branch_mode="concat",
        is_spec_norm=True,
        is_res=True,
        c_repr_mode="l1",
        c_repr_first=0,
        c_repr_base=2,
        z_mode="None",
        z_first=2,
        z_dim=4,
        pos_embed_mode="None",
        aggr_mode="sum",
        img_dims=2,
        act_name="leakyrelu0.2",
        normalization_type="None",
        dropout=0,
        self_attn_mode="None",
        last_act_name="None",
        n_avg_pool=0,
    ):
        """
        Initialize the EBM.

        Args:
            w_type: type of the first two arities of input.
                choose from "image", "mask", "image+mask", "obj", "image+obj"
        """
        super(ConceptEBM, self).__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.repr_dim = repr_dim
        self.w_type = w_type
        self.w_dim = 1 if "mask" in self.w_type else in_channels
        self.mask_mode = mask_mode
        self.channel_base = channel_base
        self.two_branch_mode = two_branch_mode

        if self.mode in ["concept"]:
            self.mask_arity = 1
        elif self.mode in ["operator"]:
            self.mask_arity = 2
        else:
            raise
        self.is_spec_norm = is_spec_norm
        self.is_res = is_res
        self.c_repr_mode = c_repr_mode
        self.c_repr_first = c_repr_first
        self.c_repr_base = c_repr_base
        self.z_mode = z_mode
        self.z_first = z_first
        self.z_dim = z_dim
        self.pos_embed_mode = pos_embed_mode
        self.aggr_mode = aggr_mode
        self.img_dims = img_dims
        self.act_name = act_name
        self.act = get_activation(act_name)
        self.normalization_type = normalization_type
        self.dropout = dropout
        assert self_attn_mode in ["None", "pixel"]
        self.self_attn_mode = self_attn_mode
        self.last_act_name = last_act_name
        self.n_avg_pool = n_avg_pool
        if img_dims == 2:
            kernel_size = 3
            padding = 1
        elif img_dims == 1:
            kernel_size = (3, 1)
            padding = (1, 0)
        else:
            raise
        if is_spec_norm in [True, "True"]:
            self.conv1 = spectral_norm(nn.Conv2d(in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base if self.mode=="concept" else channel_base//2, kernel_size, padding=padding), std=1)
        elif is_spec_norm in [False, "False"]:
            self.conv1 = nn.Conv2d(in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base if self.mode=="concept" else channel_base//2, kernel_size, padding=padding)
        elif is_spec_norm == "ws":
            self.conv1 = WSConv2d(in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base if self.mode=="concept" else channel_base//2, kernel_size, padding=padding)
        else:
            raise

        if self.mode in ["concept"] or self.two_branch_mode == "concat":
            self.blocks = nn.ModuleList(
                [
                    CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    CResBlock(channel_base, channel_base, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    CResBlock(channel_base, channel_base*2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=2 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=2 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=3 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=3 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                ]
            )
            if self_attn_mode != "None":
                assert self_attn_mode in ["pixel"]
                self.attn = Self_Attn(in_dim=channel_base, act_name=act_name)
        elif self.two_branch_mode.startswith("imbal"):
            n_indi_layers = int(self.two_branch_mode.split("-")[1])
            if self_attn_mode != "None":
                assert self_attn_mode in ["pixel"]
                self.attn_0 = Self_Attn(in_dim=channel_base//2, act_name=act_name)
                self.attn_1 = Self_Attn(in_dim=channel_base//2, act_name=act_name)
            if n_indi_layers == 1:
                self.blocks_0 = nn.ModuleList(
                    [
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    ]
                )
                self.blocks_1 = nn.ModuleList(
                    [
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    ]
                )
                self.blocks = nn.ModuleList(
                    [
                        CResBlock(channel_base, channel_base*2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=2 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=2 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=3 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=3 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    ]
                )
            elif n_indi_layers == 2:
                self.blocks_0 = nn.ModuleList(
                    [
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base, channel_base, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    ]
                )
                self.blocks_1 = nn.ModuleList(
                    [
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base, channel_base, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    ]
                )
                self.blocks = nn.ModuleList(
                    [
                        CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=2 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=2 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=3 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=3 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    ]
                )
            elif n_indi_layers == 3:
                self.blocks_0 = nn.ModuleList(
                    [
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base, channel_base, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=2 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=2 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    ]
                )
                self.blocks_1 = nn.ModuleList(
                    [
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base//2, channel_base, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base, channel_base, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                        CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=2 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=2 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    ]
                )
                self.blocks = nn.ModuleList(
                    [
                        CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, is_spec_norm=is_spec_norm, is_res=is_res, c_repr_mode=c_repr_mode if c_repr_first<=3 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=3 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout),
                    ]
                )
        else:
            raise Exception("two_branch_mode '{}' is invalid!".format(self.two_branch_mode))

        self.linear = nn.Linear(channel_base*2, 1)

    def forward(self, input, mask, c_repr=None, z=None, **kwargs):
        """
        Returns the energy of a given mask on top of a given input.

        Args:
            input: input of shape [B, C, H, W]
            mask: mask of shape [B, c, H, W]
            c_repr: shape [B, repr_dim], explicit concept representation to use for the mask. If not set, uses self.c_repr.
            z: a 1-tuple, element has shape [B, z_dim], latent representation depending on input and mask.

        Returns:
            out: energy, with shape [B, 1]
        """
        length = len(mask[0])
        if c_repr is None:
            c_repr = self.c_repr.expand(length, self.c_repr.shape[1])
        else:
            if c_repr.shape[0] == 1:
                c_repr = c_repr.expand(length, c_repr.shape[1])
        c_repr_first_dict = {0: 0, 1: 2, 2: 4, 3: 5}
        z_first_dict = {0: 0, 1: 2, 2: 4, 3: 5}
        if self.mode == "concept":
            assert len(mask) == 1
            if self.mask_mode == "concat":
                input_aug = torch.cat([input, mask[0]], 1)  # input: [B, C, H, W], mask: [B, c, H, W]
            elif self.mask_mode == "mul":
                input_aug = input * mask[0]
            elif self.mask_mode == "mulcat":
                input_aug = torch.cat([input*mask[0], mask[0]], 1)
            elif self.mask_mode == "fil":
                assert "obj" not in self.w_type
                input_aug = torch.cat([torch.clamp(input[:,:1]+(1-mask[0]), 0, 1), input[:,1:]*mask[0]], 1)
            elif self.mask_mode == "filcat":
                assert "obj" not in self.w_type
                input_fil = torch.cat([torch.clamp(input[:,:1]+(1-mask[0]), 0, 1), input[:,1:]*mask[0]], 1)
                input_aug = torch.cat([input_fil, mask[0]], 1)
            else:
                raise
            for _ in range(self.n_avg_pool):
                input_aug = F.avg_pool2d(input_aug, 3, stride=2, padding=1)
            out = self.conv1(input_aug)

            out = self.act(out)

            for i, block in enumerate(self.blocks):
                if hasattr(self, "self_attn_mode") and self.self_attn_mode != "None" and i == 2:
                    out, _ = self.attn(out)
                out = block(
                    out,
                    c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=i else None,
                    z=z if z_first_dict[self.z_first]<=i else None,
                )

            out = F.relu(out)
            if self.aggr_mode == "sum":
                out = out.view(out.shape[0], out.shape[1], -1).sum(2)
            elif self.aggr_mode == "max":
                out = out.view(out.shape[0], out.shape[1], -1).max(2)[0]
            elif self.aggr_mode == "mean":
                out = out.view(out.shape[0], out.shape[1], -1).mean(2)
            else:
                raise
            out = self.linear(out)

        elif self.mode == "operator":
            # Combining input image and mask:
            if not (isinstance(input, tuple) or isinstance(input, list)):
                input = (input, input)
            if self.mask_mode == "concat":
                input_aug_0 = torch.cat([input[0], mask[0]], 1)
                input_aug_1 = torch.cat([input[1], mask[1]], 1)
            elif self.mask_mode == "mul":
                input_aug_0 = input[0] * mask[0]
                input_aug_1 = input[1] * mask[1]
            elif self.mask_mode == "mulcat":
                input_aug_0 = torch.cat([input[0]*mask[0], mask[0]], 1)
                input_aug_1 = torch.cat([input[1]*mask[1], mask[1]], 1)
            elif self.mask_mode == "fil":
                assert "obj" not in self.w_type
                input_aug_0 = torch.cat([torch.clamp(input[0][:,:1]+(1-mask[0]), 0, 1), input[0][:,1:]*mask[0]], 1)
                input_aug_1 = torch.cat([torch.clamp(input[1][:,:1]+(1-mask[1]), 0, 1), input[1][:,1:]*mask[1]], 1)
            elif self.mask_mode == "filcat":
                assert "obj" not in self.w_type
                input_fil_0 = torch.cat([torch.clamp(input[0][:,:1]+(1-mask[0]), 0, 1), input[0][:,1:]*mask[0]], 1)
                input_aug_0 = torch.cat([input_fil_0, mask[0]], 1)
                input_fil_1 = torch.cat([torch.clamp(input[1][:,:1]+(1-mask[1]), 0, 1), input[1][:,1:]*mask[1]], 1)
                input_aug_1 = torch.cat([input_fil_1, mask[1]], 1)
            else:
                raise

            for _ in range(self.n_avg_pool):
                input_aug_0 = F.avg_pool2d(input_aug_0, 3, stride=2, padding=1)
                input_aug_1 = F.avg_pool2d(input_aug_1, 3, stride=2, padding=1)

            out_0 = self.act(self.conv1(input_aug_0))
            out_1 = self.act(self.conv1(input_aug_1))

            if self.two_branch_mode == "concat":
                out = torch.cat([out_0, out_1], 1)

                for i, block in enumerate(self.blocks):
                    if hasattr(self, "self_attn_mode") and self.self_attn_mode != "None" and i == 2:
                        out, _ = self.attn(out)
                    out = block(
                        out,
                        c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=i else None,
                        z=z if z_first_dict[self.z_first]<=i else None,
                    )

                out = F.relu(out)  # [B, 128, 1, 1]
                if self.aggr_mode == "sum":
                    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
                elif self.aggr_mode == "max":
                    out = out.view(out.shape[0], out.shape[1], -1).max(2)[0]
                elif self.aggr_mode == "mean":
                    out = out.view(out.shape[0], out.shape[1], -1).mean(2)
                else:
                    raise
                out = self.linear(out)  # [B, 1]
            elif self.two_branch_mode.startswith("imbal"):
                default_c_repr = torch.ones_like(c_repr).to(c_repr.device)
                default_z = torch.ones_like(z).to(z.device)

                for i, block_0 in enumerate(self.blocks_0):
                    if hasattr(self, "self_attn_mode") and self.self_attn_mode != "None" and i == 2:
                        out_0, _ = self.attn_0(out_0)
                    out_0 = block_0(
                        out_0,
                        c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=i else None,
                        z=z if z_first_dict[self.z_first]<=i else None,
                    )
                for i, block_1 in enumerate(self.blocks_1):
                    if hasattr(self, "self_attn_mode") and self.self_attn_mode != "None" and i == 2:
                        out_1, _ = self.attn_1(out_1)
                    out_1 = block_1(
                        out_1,
                        c_repr=default_c_repr if c_repr_first_dict[self.c_repr_first]<=i else None,
                        z=default_z if z_first_dict[self.z_first]<=i else None,
                    )
                out = torch.cat([out_0, out_1], 1)
                for i, block in enumerate(self.blocks):
                    out = block(
                        out,
                        c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=i+len(self.blocks_0) else None,
                        z=z if z_first_dict[self.z_first]<=i+len(self.blocks_0) else None,
                    )
                out = F.relu(out)
                if self.aggr_mode == "sum":
                    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
                elif self.aggr_mode == "max":
                    out = out.view(out.shape[0], out.shape[1], -1).max(2)[0]
                elif self.aggr_mode == "mean":
                    out = out.view(out.shape[0], out.shape[1], -1).mean(2)
                else:
                    raise

                out = self.linear(out)

        else:
            raise Exception("self.mode {} is not supported!".format(self.mode))

        # Last activation:
        if hasattr(self, "last_act_name") and self.last_act_name != "None":
            if self.last_act_name in ["square", "softplus", "exp", "sigmoid"]:
                out = get_activation(self.last_act_name)(out)
            else:
                raise
        return out

    def classify(self, input, mask, concept_collection, topk=-1, CONCEPTS=None, OPERATORS=None):
        """
        Given the input and mask, classify the selected concept by picking the
        lowest-energy concept from concept_collection.
        """
        if isinstance(input, tuple) or isinstance(input, list):
            length = len(input[0])
            device = input[0].device
        else:
            length = len(input)
            device = input.device
        if topk == -1:
            topk = len(concept_collection)
        c_repr_energy = []
        for j in range(len(concept_collection)):
            c_repr = id_to_tensor([concept_collection[j]] * length, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device) # len 4
            neg_energy = self(input, mask=mask, c_repr=c_repr)
            c_repr_energy.append(neg_energy)
        c_repr_energy = torch.cat(c_repr_energy, 1)
        c_repr_argsort = c_repr_energy.argsort(1)
        c_repr_pred_list = []
        for i, argsort in enumerate(c_repr_argsort):
            c_repr_pred = {}
            for k in range(min(topk, len(concept_collection))):
                id_k = c_repr_argsort[i][k]
                c_repr_pred[concept_collection[id_k]] = c_repr_energy[i][id_k].item()
            c_repr_pred_list.append(c_repr_pred)
        return c_repr_pred_list

    def ground(
        self,
        input,
        args,
        mask=None,
        c_repr=None,
        z=None,
        ensemble_size=18,
        topk=-1,
        w_init_type="random",
        sample_step=150,
        is_grad=False,
        is_return_E=False,
        isplot=True,
        **kwargs
    ):
        """
        Given an input image, find the best mask to match the given (optional)
        concept representation.

        Internally, uses a model ensemble for best results.
        """
        
        def init_neg_mask(input, init, ensemble_size):
            """Initialize negative mask"""
            if isinstance(input, tuple):
                assert len(input[0].shape) == len(input[1].shape) == 4
                device = input[0].device
                w_dim = 1 if "mask" in self.w_type else input[0].shape[1]
                neg_mask = (torch.rand(input[0].shape[0]*ensemble_size, w_dim, *input[0].shape[2:]).to(device), torch.rand(input[1].shape[0]*ensemble_size, w_dim, *input[1].shape[2:]).to(device))
                if init == "input-mask":
                    assert input[0].shape[1] == 10
                    input_l = repeat_n(input, n_repeats=ensemble_size)
                    neg_mask = (neg_mask[0] * (input_l[0].argmax(1)[:, None] != 0), neg_mask[1] * (input_l[1].argmax(1)[:, None] != 0))
                neg_mask[0].requires_grad = True
                neg_mask[1].requires_grad = True
            else:
                assert len(input.shape) == 4
                device = input.device
                w_dim = 1 if "mask" in self.w_type else input.shape[1]
                neg_mask = (torch.rand(input.shape[0]*ensemble_size, w_dim, *input.shape[2:]).to(device),)
                if init == "input-mask":
                    assert input.shape[1] == 10
                    input_l = repeat_n(input, n_repeats=ensemble_size)
                    neg_mask = (neg_mask[0] * (input_l.argmax(1)[:, None] != 0),)
                neg_mask[0].requires_grad = True
            return neg_mask

        # Update args:
        args = deepcopy(args)
        for key, value in kwargs.items():
            setattr(args, key, value)
        args.sample_step = sample_step
        if isinstance(input, tuple) or isinstance(input, list):
            args.is_image_tuple = True
            device = input[0].device
        else:
            args.is_image_tuple = False
            device = input.device

        # Perform SGLD:
        if mask is None:
            neg_mask = init_neg_mask(input, init=w_init_type, ensemble_size=ensemble_size)
        else:
            neg_mask = tuple([repeat_n(mask[0], n_repeats=ensemble_size)])

        if self.z_mode != "None":
            if z is None:
                z = tuple([torch.rand(neg_mask[0].shape[0], self.z_dim, device=device)])
            else:
                z = tuple([repeat_n(z[0], n_repeats=ensemble_size)])

        (img_ensemble, neg_mask_ensemble, z_ensemble, zgnn_ensemble, wtarget_ensemble), neg_out_list_ensemble, info_ensemble = neg_mask_sgd_ensemble(
            self, input, neg_mask, c_repr, z=z, zgnn=None, wtarget=None, args=args,
            ensemble_size=ensemble_size, is_grad=is_grad,
            is_return_E=is_return_E,
        )

        neg_out_ensemble = neg_out_list_ensemble[-1]  # neg_out_ensemble: [ensemble_size, B]
        # Sort the obtained results by energy for each example:
        neg_out_ensemble = torch.FloatTensor(neg_out_ensemble).transpose(0,1)  # [B, ensemble_size]
        neg_out_argsort = neg_out_ensemble.argsort(1)  # [B, ensemble_size]
        batch_size = neg_out_argsort.shape[0]
        neg_out_ensemble_sorted = torch.stack([neg_out_ensemble[i][neg_out_argsort[i]] for i in range(batch_size)])
        if zgnn_ensemble is not None or wtarget_ensemble is not None:
            neg_task_out_ensemble = neg_out_ensemble.reshape(*batch_shape, -1).mean(1)  # [B_task, ensemble_size]
            neg_task_out_argsort = neg_task_out_ensemble.argsort(1)

        if img_ensemble is not None:
            if args.is_image_tuple:
                img_ensemble = tuple(img_ensemble[k].transpose(0,1) for k in range(len(img)))  # Each element [B, ensemble_size, C, H, W]
                img_ensemble_sorted = []
                for k in range(len(img)):
                    img_ensemble_sorted.append(torch.stack([img_ensemble[k][i][neg_out_argsort[i]] for i in range(batch_size)]))
                img_ensemble_sorted = tuple(img_ensemble_sorted)  # each element: [B, ensemble_size, C, H, W] sorted along dim=1 according to neg_out
            else:
                img_ensemble = img_ensemble.transpose(0,1)  # [B, ensemble_size, C, H, W]
                img_ensemble_sorted = torch.stack([img_ensemble[i][neg_out_argsort[i]] for i in range(batch_size)])
        else:
            img_ensemble_sorted = None

        if neg_mask_ensemble is not None:
            neg_mask_ensemble = tuple(neg_mask_ensemble[k].transpose(0,1) for k in range(self.mask_arity))  # Each element [B, ensemble_size, C, H, W]
            neg_mask_ensemble_sorted = []
            for k in range(self.mask_arity):
                neg_mask_ensemble_sorted.append(torch.stack([neg_mask_ensemble[k][i][neg_out_argsort[i]] for i in range(len(neg_mask_ensemble[0]))]))
            neg_mask_ensemble_sorted = tuple(neg_mask_ensemble_sorted)  # each element: [B, ensemble_size, C, H, W] sorted along dim=1 according to neg_out
        else:
            neg_mask_ensemble_sorted = None

        if z_ensemble is not None:
            z_ensemble = tuple(z_ensemble[k].transpose(0,1) for k in range(len(z_ensemble)))  # Each element [B, ensemble_size, Z]
            z_ensemble_sorted = []
            for k in range(len(z_ensemble)):
                z_ensemble_sorted.append(torch.stack([z_ensemble[k][i][neg_out_argsort[i]] for i in range(batch_size)]))
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
        return (img_ensemble_sorted, neg_mask_ensemble_sorted, z_ensemble_sorted, zgnn_ensemble_sorted, wtarget_ensemble_sorted), neg_out_ensemble_sorted

    def set_c(self, c_repr, c_str=None):
        """Set default c_repr."""
        assert len(c_repr.shape) == 2 and c_repr.shape[0] == 1
        if not isinstance(c_repr, torch.Tensor):
            c_repr = torch.FloatTensor(c_repr)
        device = next(iter(self.parameters())).device
        self.c_repr = c_repr.to(device)
        self.c_str = c_str
        return self

    def set_repr_dict(self, c_repr_dict):
        """Set the dictionary of c_repr."""
        self.c_repr_dict = {}
        device = next(iter(self.parameters())).device
        for key, c_repr in c_repr_dict.items():
            if not isinstance(c_repr, torch.Tensor):
                c_repr = torch.FloatTensor(c_repr)
            self.c_repr_dict[key] = c_repr.to(device)
        return self

    def clone(self):
        """Clone the full instance."""
        return pickle.loads(pickle.dumps(self))

    def to(self, device):
        """Move to device."""
        if hasattr(self, "c_repr"):
            self.c_repr = self.c_repr.to(device)
        super().to(device)
        return self

    def __add__(self, model):
        return SumEBM(self, model)

    @property
    def model_dict(self):
        model_dict = {"type": "ConceptEBM"}
        model_dict["mode"] = self.mode
        model_dict["in_channels"] = self.in_channels
        model_dict["repr_dim"] = self.repr_dim
        model_dict["w_type"] = self.w_type
        model_dict["mask_mode"] = self.mask_mode
        model_dict["channel_base"] = self.channel_base
        model_dict["two_branch_mode"] = self.two_branch_mode
        model_dict["is_spec_norm"] = self.is_spec_norm
        model_dict["is_res"] = self.is_res
        model_dict["c_repr_mode"] = self.c_repr_mode
        model_dict["c_repr_first"] = self.c_repr_first
        model_dict["c_repr_base"] = self.c_repr_base
        model_dict["z_mode"] = self.z_mode
        model_dict["z_first"] = self.z_first
        model_dict["z_dim"] = self.z_dim
        model_dict["pos_embed_mode"] = self.pos_embed_mode
        model_dict["aggr_mode"] = self.aggr_mode
        model_dict["img_dims"] = self.img_dims
        model_dict["act_name"] = self.act_name
        model_dict["normalization_type"] = self.normalization_type
        model_dict["self_attn_mode"] = self.self_attn_mode
        model_dict["dropout"] = self.dropout
        model_dict["last_act_name"] = self.last_act_name
        model_dict["n_avg_pool"] = self.n_avg_pool
        if hasattr(self, "c_repr"):
            model_dict["c_repr"] = to_np_array(self.c_repr)
        if hasattr(self, "c_str"):
            model_dict["c_str"] = self.c_str
        model_dict["state_dict"] = to_cpu(self.state_dict())
        return model_dict


# ### 1.1.3 ConceptEBMLarge:

# In[ ]:


class ConceptEBMLarge(nn.Module):
    """
    An EBM designed specifically to find Concepts, e.g. Rect, Line, RectSolid or
        Relations, e.g. SameShape, SameColor.
    """
    
    def __init__(
        self,
        mode="concept",
        in_channels=10,
        repr_dim=4,
        w_type="image+mask",
        mask_mode="concat",
        channel_base=128,
        two_branch_mode="concat",
        is_spec_norm=True,
        is_res=False,
        c_repr_mode="l1",
        c_repr_first=0,
        c_repr_base=2,
        z_mode="None",
        z_first=2,
        z_dim=4,
        pos_embed_mode="None",
        aggr_mode="sum",
        img_dims=2,
        act_name="leakyrelu0.2",
        normalization_type="None",
        dropout=0,
        self_attn_mode="None",
        last_act_name="None",
        is_multiscale=True,
    ):
        """
        Initialize the EBM.

        Args:
            w_type: type of the first two arities of input.
                choose from "image", "mask", "image+mask", "obj", "image+obj"
        """
        super(ConceptEBMLarge, self).__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.repr_dim = repr_dim
        self.w_type = w_type
        self.w_dim = 1 if "mask" in self.w_type else in_channels
        self.mask_mode = mask_mode
        self.channel_base = channel_base
        self.two_branch_mode = two_branch_mode

        if self.mode in ["concept"]:
            self.mask_arity = 1
        elif self.mode in ["operator"]:
            self.mask_arity = 2
        else:
            raise
        self.is_spec_norm = is_spec_norm
        self.is_res = is_res
        self.c_repr_mode = c_repr_mode
        self.c_repr_first = c_repr_first
        self.c_repr_base = c_repr_base
        self.z_mode = z_mode
        self.z_first = z_first
        self.z_dim = z_dim
        self.pos_embed_mode = pos_embed_mode
        self.aggr_mode = aggr_mode
        self.img_dims = img_dims
        self.act_name = act_name
        self.act = get_activation(act_name)
        self.normalization_type = normalization_type
        self.dropout = dropout
        assert self_attn_mode in ["None", "pixel"]
        self.self_attn_mode = self_attn_mode
        self.last_act_name = last_act_name
        self.is_multiscale = is_multiscale
        if img_dims == 2:
            self.kernel_size = 3
            self.padding = 1
        elif img_dims == 1:
            self.kernel_size = (3, 1)
            self.padding = (1, 0)
        else:
            raise
        self.init_main_model()
        if self.is_multiscale:
            self.init_mid_model()
            self.init_small_model()


    def init_small_model(self):
        mask_mode = self.mask_mode
        channel_base = self.channel_base
        repr_dim = self.repr_dim
        is_spec_norm = self.is_spec_norm
        is_res = self.is_res
        c_repr_mode = self.c_repr_mode
        c_repr_first = self.c_repr_first
        c_repr_base = self.c_repr_base
        z_mode = self.z_mode
        z_dim = self.z_dim
        z_first = self.z_first
        img_dims = self.img_dims
        act_name = self.act_name
        normalization_type = self.normalization_type
        dropout = self.dropout

        if is_spec_norm in [True, "True"]:
            self.small_conv1 = spectral_norm(nn.Conv2d(self.in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base if self.mode=="concept" else channel_base//2, self.kernel_size, padding=self.padding), std=1)
        elif is_spec_norm in [False, "False"]:
            self.small_conv1 = nn.Conv2d(self.in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base if self.mode=="concept" else channel_base//2, self.kernel_size, padding=self.padding)
        elif is_spec_norm == "ws":
            self.small_conv1 = WSConv2d(self.in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base if self.mode=="concept" else channel_base//2, self.kernel_size, padding=self.padding)
        else:
            raise
        
        self.small_res_1a = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)
        self.small_res_1b = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)

        self.small_res_2a = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)
        self.small_res_2b = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv+rescale", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)

        self.small_energy_map = nn.Linear(channel_base*2, 1)


    def small_model(self, input, mask, c_repr=None, z=None):
        c_repr_first_dict = {0: 0, 1: 2, 2: 4, 3: 6}
        z_first_dict = {0: 0, 1: 2, 2: 4, 3: 6}
        if self.mode == "concept":
            assert len(mask) == 1
            if self.mask_mode == "concat":
                input_aug = torch.cat([input, mask[0]], 1)  # input: [B, C, H, W], mask: [B, c, H, W]
            elif self.mask_mode == "mul":
                input_aug = input * mask[0]
            elif self.mask_mode == "mulcat":
                input_aug = torch.cat([input*mask[0], mask[0]], 1)
            elif self.mask_mode == "fil":
                assert "obj" not in self.w_type
                input_aug = torch.cat([torch.clamp(input[:,:1]+(1-mask[0]), 0, 1), input[:,1:]*mask[0]], 1)
            elif self.mask_mode == "filcat":
                assert "obj" not in self.w_type
                input_fil = torch.cat([torch.clamp(input[:,:1]+(1-mask[0]), 0, 1), input[:,1:]*mask[0]], 1)
                input_aug = torch.cat([input_fil, mask[0]], 1)
            else:
                raise
            x = F.avg_pool2d(input_aug, 3, stride=2, padding=1)
            x = F.avg_pool2d(x, 3, stride=2, padding=1)
            x = self.act(self.small_conv1(x))

        elif self.mode == "operator":
            # Combining input image and mask:
            if not (isinstance(input, tuple) or isinstance(input, list)):
                input = (input, input)
            if self.mask_mode == "concat":
                input_aug_0 = torch.cat([input[0], mask[0]], 1)
                input_aug_1 = torch.cat([input[1], mask[1]], 1)
            elif self.mask_mode == "mul":
                input_aug_0 = input[0] * mask[0]
                input_aug_1 = input[1] * mask[1]
            elif self.mask_mode == "mulcat":
                input_aug_0 = torch.cat([input[0]*mask[0], mask[0]], 1)
                input_aug_1 = torch.cat([input[1]*mask[1], mask[1]], 1)
            elif self.mask_mode == "fil":
                assert "obj" not in self.w_type
                input_aug_0 = torch.cat([torch.clamp(input[0][:,:1]+(1-mask[0]), 0, 1), input[0][:,1:]*mask[0]], 1)
                input_aug_1 = torch.cat([torch.clamp(input[1][:,:1]+(1-mask[1]), 0, 1), input[1][:,1:]*mask[1]], 1)
            elif self.mask_mode == "filcat":
                assert "obj" not in self.w_type
                input_fil_0 = torch.cat([torch.clamp(input[0][:,:1]+(1-mask[0]), 0, 1), input[0][:,1:]*mask[0]], 1)
                input_aug_0 = torch.cat([input_fil_0, mask[0]], 1)
                input_fil_1 = torch.cat([torch.clamp(input[1][:,:1]+(1-mask[1]), 0, 1), input[1][:,1:]*mask[1]], 1)
                input_aug_1 = torch.cat([input_fil_1, mask[1]], 1)
            else:
                raise

            x_0 = F.avg_pool2d(input_aug_0, 3, stride=2, padding=1)
            x_0 = F.avg_pool2d(x_0, 3, stride=2, padding=1)
            x_0 = self.act(self.small_conv1(x_0))
            x_1 = F.avg_pool2d(input_aug_1, 3, stride=2, padding=1)
            x_1 = F.avg_pool2d(x_1, 3, stride=2, padding=1)
            x_1 = self.act(self.small_conv1(x_1))
            x = torch.cat([x_0, x_1], 1)

        x = self.small_res_1a(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=0 else None, z=z if z_first_dict[self.z_first]<=0 else None)
        x = self.small_res_1b(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=1 else None, z=z if z_first_dict[self.z_first]<=1 else None)

        x = self.small_res_2a(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=2 else None, z=z if z_first_dict[self.z_first]<=2 else None)
        x = self.small_res_2b(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=3 else None, z=z if z_first_dict[self.z_first]<=3 else None)
        x = self.act(x)

        if self.aggr_mode == "sum":
            x = x.view(x.shape[0], x.shape[1], -1).sum(2)
        elif self.aggr_mode == "max":
            x = x.view(x.shape[0], x.shape[1], -1).max(2)[0]
        elif self.aggr_mode == "mean":
            x = x.view(x.shape[0], x.shape[1], -1).mean(2)
        else:
            raise

        energy = self.small_energy_map(x)

        # Last activation:
        if self.last_act_name != "None":
            if self.last_act_name in ["square", "softplus", "exp", "sigmoid"]:
                energy = get_activation(self.last_act_name)(energy)
            else:
                raise
        return energy


    def init_mid_model(self):
        mask_mode = self.mask_mode
        channel_base = self.channel_base
        repr_dim = self.repr_dim
        is_spec_norm = self.is_spec_norm
        is_res = self.is_res
        c_repr_mode = self.c_repr_mode
        c_repr_first = self.c_repr_first
        c_repr_base = self.c_repr_base
        z_mode = self.z_mode
        z_dim = self.z_dim
        z_first = self.z_first
        img_dims = self.img_dims
        act_name = self.act_name
        normalization_type = self.normalization_type
        dropout = self.dropout

        if is_spec_norm in [True, "True"]:
            self.mid_conv1 = spectral_norm(nn.Conv2d(self.in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base if self.mode=="concept" else channel_base//2, self.kernel_size, padding=self.padding), std=1)
        elif is_spec_norm in [False, "False"]:
            self.mid_conv1 = nn.Conv2d(self.in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base if self.mode=="concept" else channel_base//2, self.kernel_size, padding=self.padding)
        elif is_spec_norm == "ws":
            self.mid_conv1 = WSConv2d(self.in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base if self.mode=="concept" else channel_base//2, self.kernel_size, padding=self.padding)
        else:
            raise

        self.mid_res_1a = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)
        self.mid_res_1b = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)

        self.mid_res_2a = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)
        self.mid_res_2b = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv+rescale", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)

        self.mid_res_3a = CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, downsample=False, is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=2 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=2 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)
        self.mid_res_3b = CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, downsample=True, downsample_mode="conv+rescale", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=2 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=2 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)

        self.mid_energy_map = nn.Linear(channel_base*4, 1)
        self.avg_pool = Downsample(channels=3)


    def mid_model(self, input, mask, c_repr=None, z=None):
        c_repr_first_dict = {0: 0, 1: 2, 2: 4, 3: 6}
        z_first_dict = {0: 0, 1: 2, 2: 4, 3: 6}
        if self.mode == "concept":
            assert len(mask) == 1
            if self.mask_mode == "concat":
                input_aug = torch.cat([input, mask[0]], 1)  # input: [B, C, H, W], mask: [B, c, H, W]
            elif self.mask_mode == "mul":
                input_aug = input * mask[0]
            elif self.mask_mode == "mulcat":
                input_aug = torch.cat([input*mask[0], mask[0]], 1)
            elif self.mask_mode == "fil":
                assert "obj" not in self.w_type
                input_aug = torch.cat([torch.clamp(input[:,:1]+(1-mask[0]), 0, 1), input[:,1:]*mask[0]], 1)
            elif self.mask_mode == "filcat":
                assert "obj" not in self.w_type
                input_fil = torch.cat([torch.clamp(input[:,:1]+(1-mask[0]), 0, 1), input[:,1:]*mask[0]], 1)
                input_aug = torch.cat([input_fil, mask[0]], 1)
            else:
                raise
            x = F.avg_pool2d(input_aug, 3, stride=2, padding=1)
            x = self.act(self.mid_conv1(x))
        
        elif self.mode == "operator":
            # Combining input image and mask:
            if not (isinstance(input, tuple) or isinstance(input, list)):
                input = (input, input)
            if self.mask_mode == "concat":
                input_aug_0 = torch.cat([input[0], mask[0]], 1)
                input_aug_1 = torch.cat([input[1], mask[1]], 1)
            elif self.mask_mode == "mul":
                input_aug_0 = input[0] * mask[0]
                input_aug_1 = input[1] * mask[1]
            elif self.mask_mode == "mulcat":
                input_aug_0 = torch.cat([input[0]*mask[0], mask[0]], 1)
                input_aug_1 = torch.cat([input[1]*mask[1], mask[1]], 1)
            elif self.mask_mode == "fil":
                assert "obj" not in self.w_type
                input_aug_0 = torch.cat([torch.clamp(input[0][:,:1]+(1-mask[0]), 0, 1), input[0][:,1:]*mask[0]], 1)
                input_aug_1 = torch.cat([torch.clamp(input[1][:,:1]+(1-mask[1]), 0, 1), input[1][:,1:]*mask[1]], 1)
            elif self.mask_mode == "filcat":
                assert "obj" not in self.w_type
                input_fil_0 = torch.cat([torch.clamp(input[0][:,:1]+(1-mask[0]), 0, 1), input[0][:,1:]*mask[0]], 1)
                input_aug_0 = torch.cat([input_fil_0, mask[0]], 1)
                input_fil_1 = torch.cat([torch.clamp(input[1][:,:1]+(1-mask[1]), 0, 1), input[1][:,1:]*mask[1]], 1)
                input_aug_1 = torch.cat([input_fil_1, mask[1]], 1)
            else:
                raise

            x_0 = F.avg_pool2d(input_aug_0, 3, stride=2, padding=1)
            x_0 = self.act(self.mid_conv1(x_0))
            x_1 = F.avg_pool2d(input_aug_1, 3, stride=2, padding=1)
            x_1 = self.act(self.mid_conv1(x_1))
            x = torch.cat([x_0, x_1], 1)

        x = self.mid_res_1a(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=0 else None, z=z if z_first_dict[self.z_first]<=0 else None)
        x = self.mid_res_1b(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=1 else None, z=z if z_first_dict[self.z_first]<=1 else None)

        x = self.mid_res_2a(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=2 else None, z=z if z_first_dict[self.z_first]<=2 else None)
        x = self.mid_res_2b(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=3 else None, z=z if z_first_dict[self.z_first]<=3 else None)

        x = self.mid_res_3a(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=4 else None, z=z if z_first_dict[self.z_first]<=4 else None)
        x = self.mid_res_3b(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=5 else None, z=z if z_first_dict[self.z_first]<=5 else None)
        x = self.act(x)

        if self.aggr_mode == "sum":
            x = x.view(x.shape[0], x.shape[1], -1).sum(2)
        elif self.aggr_mode == "max":
            x = x.view(x.shape[0], x.shape[1], -1).max(2)[0]
        elif self.aggr_mode == "mean":
            x = x.view(x.shape[0], x.shape[1], -1).mean(2)
        else:
            raise

        energy = self.mid_energy_map(x)

        # Last activation:
        if self.last_act_name != "None":
            if self.last_act_name in ["square", "softplus", "exp", "sigmoid"]:
                energy = get_activation(self.last_act_name)(energy)
            else:
                raise
        return energy


    def init_main_model(self):
        mask_mode = self.mask_mode
        channel_base = self.channel_base
        repr_dim = self.repr_dim
        is_spec_norm = self.is_spec_norm
        is_res = self.is_res
        c_repr_mode = self.c_repr_mode
        c_repr_first = self.c_repr_first
        c_repr_base = self.c_repr_base
        z_mode = self.z_mode
        z_dim = self.z_dim
        z_first = self.z_first
        img_dims = self.img_dims
        act_name = self.act_name
        normalization_type = self.normalization_type
        dropout = self.dropout

        if is_spec_norm in [True, "True"]:
            self.conv1 = spectral_norm(nn.Conv2d(self.in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base//2 if self.mode=="concept" else channel_base//4, self.kernel_size, padding=self.padding), std=1)
        elif is_spec_norm in [False, "False"]:
            self.conv1 = nn.Conv2d(self.in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base//2 if self.mode=="concept" else channel_base//4, self.kernel_size, padding=self.padding)
        elif is_spec_norm == "ws":
            self.conv1 = WSConv2d(self.in_channels+self.w_dim if mask_mode in ["concat", "mulcat", "filcat"] else in_channels, channel_base//2 if self.mode=="concept" else channel_base//4, self.kernel_size, padding=self.padding)
        else:
            raise

        self.res_1a = CResBlock(channel_base//2, channel_base//2, repr_dim=repr_dim, downsample=True, downsample_mode="conv+rescale", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)
        self.res_1b = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=0 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=0 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)

        self.res_2a = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)
        self.res_2b = CResBlock(channel_base, channel_base, repr_dim=repr_dim, downsample=True, downsample_mode="conv+rescale", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=1 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=1 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)

        self.res_3a = CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, downsample=False, is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=2 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=2 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)
        self.res_3b = CResBlock(channel_base*2, channel_base*2, repr_dim=repr_dim, downsample=True, downsample_mode="conv+rescale", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=2 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=2 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)

        self.res_4a = CResBlock(channel_base*4, channel_base*4, repr_dim=repr_dim, downsample=False, is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=3 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=3 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)
        self.res_4b = CResBlock(channel_base*4, channel_base*4, repr_dim=repr_dim, downsample=True, downsample_mode="conv+rescale", is_res=is_res, is_spec_norm=is_spec_norm, c_repr_mode=c_repr_mode if c_repr_first<=3 else "None", c_repr_base=c_repr_base, z_mode=z_mode if z_first<=3 else "None", z_dim=z_dim, img_dims=img_dims, act_name=act_name, normalization_type=normalization_type, dropout=dropout)

        if self.self_attn_mode in ["pixel"]:
            self.self_attn = Self_Attn(4 * channel_base, self.act_name)

        self.energy_map = nn.Linear(channel_base*8, 1)


    def main_model(self, input, mask, c_repr=None, z=None):
        c_repr_first_dict = {0: 0, 1: 2, 2: 4, 3: 6}
        z_first_dict = {0: 0, 1: 2, 2: 4, 3: 6}
        if self.mode == "concept":
            assert len(mask) == 1
            if self.mask_mode == "concat":
                input_aug = torch.cat([input, mask[0]], 1)  # input: [B, C, H, W], mask: [B, c, H, W]
            elif self.mask_mode == "mul":
                input_aug = input * mask[0]
            elif self.mask_mode == "mulcat":
                input_aug = torch.cat([input*mask[0], mask[0]], 1)
            elif self.mask_mode == "fil":
                assert "obj" not in self.w_type
                input_aug = torch.cat([torch.clamp(input[:,:1]+(1-mask[0]), 0, 1), input[:,1:]*mask[0]], 1)
            elif self.mask_mode == "filcat":
                assert "obj" not in self.w_type
                input_fil = torch.cat([torch.clamp(input[:,:1]+(1-mask[0]), 0, 1), input[:,1:]*mask[0]], 1)
                input_aug = torch.cat([input_fil, mask[0]], 1)
            else:
                raise
            x = self.act(self.conv1(input_aug))

        elif self.mode == "operator":
            # Combining input image and mask:
            if not (isinstance(input, tuple) or isinstance(input, list)):
                input = (input, input)
            if self.mask_mode == "concat":
                input_aug_0 = torch.cat([input[0], mask[0]], 1)
                input_aug_1 = torch.cat([input[1], mask[1]], 1)
            elif self.mask_mode == "mul":
                input_aug_0 = input[0] * mask[0]
                input_aug_1 = input[1] * mask[1]
            elif self.mask_mode == "mulcat":
                input_aug_0 = torch.cat([input[0]*mask[0], mask[0]], 1)
                input_aug_1 = torch.cat([input[1]*mask[1], mask[1]], 1)
            elif self.mask_mode == "fil":
                assert "obj" not in self.w_type
                input_aug_0 = torch.cat([torch.clamp(input[0][:,:1]+(1-mask[0]), 0, 1), input[0][:,1:]*mask[0]], 1)
                input_aug_1 = torch.cat([torch.clamp(input[1][:,:1]+(1-mask[1]), 0, 1), input[1][:,1:]*mask[1]], 1)
            elif self.mask_mode == "filcat":
                assert "obj" not in self.w_type
                input_fil_0 = torch.cat([torch.clamp(input[0][:,:1]+(1-mask[0]), 0, 1), input[0][:,1:]*mask[0]], 1)
                input_aug_0 = torch.cat([input_fil_0, mask[0]], 1)
                input_fil_1 = torch.cat([torch.clamp(input[1][:,:1]+(1-mask[1]), 0, 1), input[1][:,1:]*mask[1]], 1)
                input_aug_1 = torch.cat([input_fil_1, mask[1]], 1)
            else:
                raise

            x_0 = self.act(self.conv1(input_aug_0))
            x_1 = self.act(self.conv1(input_aug_1))
            x = torch.cat([x_0, x_1], 1)

        x = self.res_1a(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=0 else None, z=z if z_first_dict[self.z_first]<=0 else None)
        x = self.res_1b(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=1 else None, z=z if z_first_dict[self.z_first]<=1 else None)

        x = self.res_2a(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=2 else None, z=z if z_first_dict[self.z_first]<=2 else None)
        x = self.res_2b(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=3 else None, z=z if z_first_dict[self.z_first]<=3 else None)

        x = self.res_3a(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=4 else None, z=z if z_first_dict[self.z_first]<=4 else None)
        x = self.res_3b(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=5 else None, z=z if z_first_dict[self.z_first]<=5 else None)

        if self.self_attn_mode in ["pixel"]:
            x, _ = self.self_attn(x)

        x = self.res_4a(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=6 else None, z=z if z_first_dict[self.z_first]<=6 else None)
        x = self.res_4b(x, c_repr=c_repr if c_repr_first_dict[self.c_repr_first]<=7 else None, z=z if z_first_dict[self.z_first]<=7 else None)
        x = self.act(x)

        if self.aggr_mode == "sum":
            x = x.view(x.shape[0], x.shape[1], -1).sum(2)
        elif self.aggr_mode == "max":
            x = x.view(x.shape[0], x.shape[1], -1).max(2)[0]
        elif self.aggr_mode == "mean":
            x = x.view(x.shape[0], x.shape[1], -1).mean(2)
        else:
            raise

        energy = self.energy_map(x)

        # Last activation:
        if self.last_act_name != "None":
            if self.last_act_name in ["square", "softplus", "exp", "sigmoid"]:
                energy = get_activation(self.last_act_name)(energy)
            else:
                raise
        return energy


    def forward(self, input, mask, c_repr=None, z=None, **kwargs):
        energy = self.main_model(input, mask, c_repr=c_repr, z=z)

        if self.is_multiscale:
            large_energy = energy
            mid_energy = self.mid_model(input, mask, c_repr=c_repr, z=z)
            small_energy = self.small_model(input, mask, c_repr=c_repr, z=z)
            energy = torch.cat([small_energy, mid_energy, large_energy], dim=-1).sum(-1, keepdims=True)
        return energy


    def classify(self, input, mask, concept_collection, topk=-1, CONCEPTS=None, OPERATORS=None):
        """
        Given the input and mask, classify the selected concept by picking the
        lowest-energy concept from concept_collection.
        """
        if isinstance(input, tuple) or isinstance(input, list):
            length = len(input[0])
            device = input[0].device
        else:
            length = len(input)
            device = input.device
        if topk == -1:
            topk = len(concept_collection)
        c_repr_energy = []
        for j in range(len(concept_collection)):
            c_repr = id_to_tensor([concept_collection[j]] * length, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device) # len 4
            neg_energy = self(input, mask=mask, c_repr=c_repr)
            c_repr_energy.append(neg_energy)
        c_repr_energy = torch.cat(c_repr_energy, 1)
        c_repr_argsort = c_repr_energy.argsort(1)
        c_repr_pred_list = []
        for i, argsort in enumerate(c_repr_argsort):
            c_repr_pred = {}
            for k in range(min(topk, len(concept_collection))):
                id_k = c_repr_argsort[i][k]
                c_repr_pred[concept_collection[id_k]] = c_repr_energy[i][id_k].item()
            c_repr_pred_list.append(c_repr_pred)
        return c_repr_pred_list


    def ground(
        self,
        input,
        args,
        mask=None,
        c_repr=None,
        z=None,
        ensemble_size=18,
        topk=-1,
        w_init_type="random",
        sample_step=150,
        is_grad=False,
        is_return_E=False,
        isplot=True,
        **kwargs
    ):
        """
        Given an input image, find the best mask to match the given (optional)
        concept representation.

        Internally, uses a model ensemble for best results.
        """
        
        def init_neg_mask(input, init, ensemble_size):
            """Initialize negative mask"""
            if isinstance(input, tuple):
                assert len(input[0].shape) == len(input[1].shape) == 4
                device = input[0].device
                w_dim = 1 if "mask" in self.w_type else input[0].shape[1]
                neg_mask = (torch.rand(input[0].shape[0]*ensemble_size, w_dim, *input[0].shape[2:]).to(device), torch.rand(input[1].shape[0]*ensemble_size, w_dim, *input[1].shape[2:]).to(device))
                if init == "input-mask":
                    assert input[0].shape[1] == 10
                    input_l = repeat_n(input, n_repeats=ensemble_size)
                    neg_mask = (neg_mask[0] * (input_l[0].argmax(1)[:, None] != 0), neg_mask[1] * (input_l[1].argmax(1)[:, None] != 0))
                neg_mask[0].requires_grad = True
                neg_mask[1].requires_grad = True
            else:
                assert len(input.shape) == 4
                device = input.device
                w_dim = 1 if "mask" in self.w_type else input.shape[1]
                neg_mask = (torch.rand(input.shape[0]*ensemble_size, w_dim, *input.shape[2:]).to(device),)
                if init == "input-mask":
                    assert input.shape[1] == 10
                    input_l = repeat_n(input, n_repeats=ensemble_size)
                    neg_mask = (neg_mask[0] * (input_l.argmax(1)[:, None] != 0),)
                neg_mask[0].requires_grad = True
            return neg_mask

        # Update args:
        args = deepcopy(args)
        for key, value in kwargs.items():
            setattr(args, key, value)
        args.sample_step = sample_step
        if isinstance(input, tuple) or isinstance(input, list):
            args.is_image_tuple = True
            device = input[0].device
        else:
            args.is_image_tuple = False
            device = input.device

        # Perform SGLD:
        if mask is None:
            neg_mask = init_neg_mask(input, init=w_init_type, ensemble_size=ensemble_size)
        else:
            neg_mask = tuple([repeat_n(mask[0], n_repeats=ensemble_size)])

        if self.z_mode != "None":
            if z is None:
                z = tuple([torch.rand(neg_mask[0].shape[0], self.z_dim, device=device)])
            else:
                z = tuple([repeat_n(z[0], n_repeats=ensemble_size)])

        (img_ensemble, neg_mask_ensemble, z_ensemble, zgnn_ensemble, wtarget_ensemble), neg_out_list_ensemble, info_ensemble = neg_mask_sgd_ensemble(
            self, input, neg_mask, c_repr, z=z, zgnn=None, wtarget=None, args=args,
            ensemble_size=ensemble_size, is_grad=is_grad,
            is_return_E=is_return_E,
        )

        neg_out_ensemble = neg_out_list_ensemble[-1]  # neg_out_ensemble: [ensemble_size, B]
        # Sort the obtained results by energy for each example:
        neg_out_ensemble = torch.FloatTensor(neg_out_ensemble).transpose(0,1)  # [B, ensemble_size]
        neg_out_argsort = neg_out_ensemble.argsort(1)  # [B, ensemble_size]
        batch_size = neg_out_argsort.shape[0]
        neg_out_ensemble_sorted = torch.stack([neg_out_ensemble[i][neg_out_argsort[i]] for i in range(batch_size)])
        if zgnn_ensemble is not None or wtarget_ensemble is not None:
            neg_task_out_ensemble = neg_out_ensemble.reshape(*batch_shape, -1).mean(1)  # [B_task, ensemble_size]
            neg_task_out_argsort = neg_task_out_ensemble.argsort(1)

        if img_ensemble is not None:
            if args.is_image_tuple:
                img_ensemble = tuple(img_ensemble[k].transpose(0,1) for k in range(len(img)))  # Each element [B, ensemble_size, C, H, W]
                img_ensemble_sorted = []
                for k in range(len(img)):
                    img_ensemble_sorted.append(torch.stack([img_ensemble[k][i][neg_out_argsort[i]] for i in range(batch_size)]))
                img_ensemble_sorted = tuple(img_ensemble_sorted)  # each element: [B, ensemble_size, C, H, W] sorted along dim=1 according to neg_out
            else:
                img_ensemble = img_ensemble.transpose(0,1)  # [B, ensemble_size, C, H, W]
                img_ensemble_sorted = torch.stack([img_ensemble[i][neg_out_argsort[i]] for i in range(batch_size)])
        else:
            img_ensemble_sorted = None

        if neg_mask_ensemble is not None:
            neg_mask_ensemble = tuple(neg_mask_ensemble[k].transpose(0,1) for k in range(self.mask_arity))  # Each element [B, ensemble_size, C, H, W]
            neg_mask_ensemble_sorted = []
            for k in range(self.mask_arity):
                neg_mask_ensemble_sorted.append(torch.stack([neg_mask_ensemble[k][i][neg_out_argsort[i]] for i in range(len(neg_mask_ensemble[0]))]))
            neg_mask_ensemble_sorted = tuple(neg_mask_ensemble_sorted)  # each element: [B, ensemble_size, C, H, W] sorted along dim=1 according to neg_out
        else:
            neg_mask_ensemble_sorted = None

        if z_ensemble is not None:
            z_ensemble = tuple(z_ensemble[k].transpose(0,1) for k in range(len(z_ensemble)))  # Each element [B, ensemble_size, Z]
            z_ensemble_sorted = []
            for k in range(len(z_ensemble)):
                z_ensemble_sorted.append(torch.stack([z_ensemble[k][i][neg_out_argsort[i]] for i in range(batch_size)]))
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
        return (img_ensemble_sorted, neg_mask_ensemble_sorted, z_ensemble_sorted, zgnn_ensemble_sorted, wtarget_ensemble_sorted), neg_out_ensemble_sorted

    def set_c(self, c_repr, c_str=None):
        """Set default c_repr."""
        assert len(c_repr.shape) == 2 and c_repr.shape[0] == 1
        if not isinstance(c_repr, torch.Tensor):
            c_repr = torch.FloatTensor(c_repr)
        device = next(iter(self.parameters())).device
        self.c_repr = c_repr.to(device)
        self.c_str = c_str
        return self

    def set_repr_dict(self, c_repr_dict):
        """Set the dictionary of c_repr."""
        self.c_repr_dict = {}
        device = next(iter(self.parameters())).device
        for key, c_repr in c_repr_dict.items():
            if not isinstance(c_repr, torch.Tensor):
                c_repr = torch.FloatTensor(c_repr)
            self.c_repr_dict[key] = c_repr.to(device)
        return self

    def clone(self):
        """Clone the full instance."""
        return pickle.loads(pickle.dumps(self))

    def to(self, device):
        """Move to device."""
        if hasattr(self, "c_repr"):
            self.c_repr = self.c_repr.to(device)
        super().to(device)
        return self

    def __add__(self, model):
        return SumEBM(self, model)

    @property
    def model_dict(self):
        model_dict = {"type": "ConceptEBMLarge"}
        model_dict["mode"] = self.mode
        model_dict["in_channels"] = self.in_channels
        model_dict["repr_dim"] = self.repr_dim
        model_dict["w_type"] = self.w_type
        model_dict["mask_mode"] = self.mask_mode
        model_dict["channel_base"] = self.channel_base
        model_dict["two_branch_mode"] = self.two_branch_mode
        model_dict["is_spec_norm"] = self.is_spec_norm
        model_dict["is_res"] = self.is_res
        model_dict["c_repr_mode"] = self.c_repr_mode
        model_dict["c_repr_first"] = self.c_repr_first
        model_dict["c_repr_base"] = self.c_repr_base
        model_dict["z_mode"] = self.z_mode
        model_dict["z_first"] = self.z_first
        model_dict["z_dim"] = self.z_dim
        model_dict["pos_embed_mode"] = self.pos_embed_mode
        model_dict["aggr_mode"] = self.aggr_mode
        model_dict["img_dims"] = self.img_dims
        model_dict["act_name"] = self.act_name
        model_dict["normalization_type"] = self.normalization_type
        model_dict["self_attn_mode"] = self.self_attn_mode
        model_dict["dropout"] = self.dropout
        model_dict["last_act_name"] = self.last_act_name
        model_dict["is_multiscale"] = self.is_multiscale
        if hasattr(self, "c_repr"):
            model_dict["c_repr"] = to_np_array(self.c_repr)
        if hasattr(self, "c_str"):
            model_dict["c_str"] = self.c_str
        model_dict["state_dict"] = to_cpu(self.state_dict())
        return model_dict


# In[ ]:


class CelebAModel(nn.Module):
    """From https://github.com/yilundu/improved_contrastive_divergence/blob/master/models.py#L413"""
    def __init__(self, args, debug=False):
        from improved_contrastive_divergence.models import CondResBlock
        super(CelebAModel, self).__init__()
        self.act = swish
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cond = args.cond

        self.args = args
        self.init_main_model()

        if args.multiscale:
            self.init_mid_model()
            self.init_small_model()

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = Downsample(channels=3)
        self.heir_weight = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0]))
        self.debug = debug

    def init_main_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.conv1 = nn.Conv2d(3, filter_dim // 2, kernel_size=3, stride=1, padding=1)

        self.res_1a = CondResBlock(args, filters=filter_dim // 2, latent_dim=latent_dim, im_size=im_size, downsample=True, classes=2, norm=args.norm, spec_norm=args.spec_norm)
        self.res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False, classes=2, norm=args.norm, spec_norm=args.spec_norm)

        self.res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False, classes=2, norm=args.norm, spec_norm=args.spec_norm)
        self.res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, classes=2, norm=args.norm, spec_norm=args.spec_norm)

        self.res_3a = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, classes=2, norm=args.norm, spec_norm=args.spec_norm)
        self.res_3b = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, classes=2, norm=args.norm, spec_norm=args.spec_norm)

        self.res_4a = CondResBlock(args, filters=4*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, classes=2, norm=args.norm, spec_norm=args.spec_norm)
        self.res_4b = CondResBlock(args, filters=4*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, classes=2, norm=args.norm, spec_norm=args.spec_norm)

        self.self_attn = Self_Attn(4 * filter_dim, self.act)

        self.energy_map = nn.Linear(filter_dim*8, 1)

    def init_mid_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.mid_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)

        self.mid_res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False, classes=2)
        self.mid_res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False, classes=2)

        self.mid_res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False, classes=2)
        self.mid_res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, classes=2)

        self.mid_res_3a = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, classes=2)
        self.mid_res_3b = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, classes=2)

        self.mid_energy_map = nn.Linear(filter_dim*4, 1)
        self.avg_pool = Downsample(channels=3)

    def init_small_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.small_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)

        self.small_res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False, classes=2)
        self.small_res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False, classes=2)

        self.small_res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False, classes=2)
        self.small_res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, classes=2)

        self.small_energy_map = nn.Linear(filter_dim*2, 1)

    def main_model(self, x, latent):
        x = self.act(self.conv1(x))

        x = self.res_1a(x, latent)
        x = self.res_1b(x, latent)

        x = self.res_2a(x, latent)
        x = self.res_2b(x, latent)


        x = self.res_3a(x, latent)
        x = self.res_3b(x, latent)

        if self.args.self_attn:
            x, _ = self.self_attn(x)

        x = self.res_4a(x, latent)
        x = self.res_4b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        energy = self.energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        if self.args.sigmoid:
            energy = F.sigmoid(energy)

        return energy

    def mid_model(self, x, latent):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.mid_conv1(x))

        x = self.mid_res_1a(x, latent)
        x = self.mid_res_1b(x, latent)

        x = self.mid_res_2a(x, latent)
        x = self.mid_res_2b(x, latent)

        x = self.mid_res_3a(x, latent)
        x = self.mid_res_3b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        energy = self.mid_energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        if self.args.sigmoid:
            energy = F.sigmoid(energy)

        return energy

    def small_model(self, x, latent):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.small_conv1(x))

        x = self.small_res_1a(x, latent)
        x = self.small_res_1b(x, latent)

        x = self.small_res_2a(x, latent)
        x = self.small_res_2b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        energy = self.small_energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        if self.args.sigmoid:
            energy = F.sigmoid(energy)

        return energy

    def label_map(self, latent):
        x = self.act(self.map_fc1(latent))
        x = self.act(self.map_fc2(x))
        x = self.act(self.map_fc3(x))
        x = self.act(self.map_fc4(x))

        return x

    def forward(self, x, latent):
        args = self.args

        if not self.cond:
            latent = None

        energy = self.main_model(x, latent)

        if args.multiscale:
            large_energy = energy
            mid_energy = self.mid_model(x, latent)
            small_energy = self.small_model(x, latent)
            energy = torch.cat([small_energy, mid_energy, large_energy], dim=-1)

        return energy


# ### 1.1.4 CEBM:

# In[ ]:


class CEBM(nn.Module):
    """
    A generic class of CEBM. From "Wu, Hao, et al. "Conjugate Energy-Based Models." ICML 2021 
    """
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__()
        self.device = device
        self.flatten = nn.Flatten()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=False, **kwargs)
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        self.nss1_net = nn.Linear(cnn_output_dim, latent_dim)
        self.nss2_net = nn.Linear(cnn_output_dim, latent_dim)

    def forward(self, x):
        h = self.flatten(self.conv_net(x))
        nss1 = self.nss1_net(h) 
        nss2 = self.nss2_net(h)
        return nss1, -nss2**2

    def log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  

    def nats_to_params(self, nat1, nat2):
        """
        convert a Gaussian natural parameters its distritbuion parameters,
        mu = - 0.5 *  (nat1 / nat2), 
        sigma = (- 0.5 / nat2).sqrt()
        nat1 : natural parameter which correspond to x,
        nat2 : natural parameter which correspond to x^2.      
        """
        mu = - 0.5 * nat1 / nat2
        sigma = (- 0.5 / nat2).sqrt()
        return mu, sigma

    def params_to_nats(self, mu, sigma):
        """
        convert a Gaussian distribution parameters to the natrual parameters
        nat1 = mean / sigma**2, 
        nat2 = - 1 / (2 * sigma**2)
        nat1 : natural parameter which correspond to x,
        nat2 : natural parameter which correspond to x^2.
        """
        nat1 = mu / (sigma**2)
        nat2 = - 0.5 / (sigma**2)
        return nat1, nat2    

    def log_factor(self, x, latents, expand_dim=None):
        """
        compute the log factor log p(x | z) for the CEBM
        """
        nss1, nss2 = self.forward(x)
        if expand_dim is not None:
            nss1 = nss1.expand(expand_dim , -1, -1)
            nss2 = nss2.expand(expand_dim , -1, -1)
            return (nss1 * latents).sum(2) + (nss2 * (latents**2)).sum(2)
        else:
            return (nss1 * latents).sum(1) + (nss2 * (latents**2)).sum(1) 

    def energy(self, x):
        pass

    def latent_params(self, x):
        pass

    def log_prior(self, latents):
        pass   


class CEBM_Gaussian(CEBM):
    """
    conjugate EBM with a spherical Gaussian inductive bias
    """
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs)
        self.ib_mean = torch.zeros(latent_dim, device=self.device)
        self.ib_log_std = torch.zeros(latent_dim, device=self.device)

    def energy(self, x):
        """
        return the energy of an input x
        """
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_mean, self.ib_log_std.exp())
        logA_prior = self.log_partition(ib_nat1, ib_nat2)
        logA_posterior = self.log_partition(ib_nat1+nss1, ib_nat2+nss2)
        return logA_prior.sum(0) - logA_posterior.sum(1)   

    def latent_params(self, x):
        """
        return the posterior distribution parameters
        """
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_mean, self.ib_log_std.exp()) 
        return self.nats_to_params(ib_nat1+nss1, ib_nat2+nss2) 

class CEBM_GMM(CEBM):
    """
    conjugate EBM with a GMM inductive bias
    """
    def __init__(self, optimize_ib, num_clusters, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs)
        #Suggested initialization
        self.ib_means = torch.randn((num_clusters, latent_dim), device=self.device)
        self.ib_log_stds = torch.ones((num_clusters, latent_dim), device=self.device).log()
        if optimize_ib:
            self.ib_means = nn.Parameter(self.ib_means)
            self.ib_log_stds = nn.Parameter(self.ib_log_stds)
        self.K = num_clusters
        self.log_K = torch.tensor([self.K], device=self.device).log()

    def energy(self, x):
        """
        return the energy of an input x
        """
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_means, self.ib_log_stds.exp())
        logA_prior = self.log_partition(ib_nat1, ib_nat2) # K * D
        logA_posterior = self.log_partition(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1)) # B * K * D
        assert logA_prior.shape == (self.K, nss1.shape[1]), 'unexpected shape.'
        assert logA_posterior.shape == (nss1.shape[0], self.K, nss1.shape[-1]), 'unexpected shape.'
        return self.log_K - torch.logsumexp(logA_posterior.sum(2) - logA_prior.sum(1), dim=-1)   

    def latent_params(self, x):
        """
        return the posterior distribution parameters
        """
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_means, self.ib_log_stds.exp())
        logA_prior = self.log_partition(ib_nat1, ib_nat2) # K * D
        logA_posterior = self.log_partition(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1)) # B * K * D
        probs = torch.nn.functional.softmax(logA_posterior.sum(2) - logA_prior.sum(1), dim=-1)
        means, stds = self.nats_to_params(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1))
        pred_y_expand = probs.argmax(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, means.shape[2])
        return torch.gather(means, 1, pred_y_expand).squeeze(1), torch.gather(stds, 1, pred_y_expand).squeeze(1)


def cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act, batchnorm, **kwargs):
    """
    building blocks for a convnet.
    each block is in form of:
        Conv2d
        BatchNorm2d(optinal)
        Activation
        Dropout(optional)
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=kwargs['leak_slope'], inplace=True)
    else:
        act = getattr(nn, activation)()
    assert len(channels) == len(kernels), "length of channels: %s,  length of kernels: %s" % (len(channels), len(kernels))
    assert len(channels) == len(strides), "length of channels: %s,  length of strides: %s" % (len(channels), len(strides))
    assert len(channels) == len(paddings), "length of channels: %s,  length of kernels: %s" % (len(channels), len(paddings))
    layers = []
    in_c = input_channels
    for i, out_c in enumerate(channels):
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        if (i < (len(channels)-1)) or last_act:#Last layer will be customized 
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(act)
            if 'dropout_prob' in kwargs:
                layers.append(nn.Dropout2d(kwargs['dropout_prob']))
            if 'maxpool_kernels' in kwargs and 'maxpool_strides' in kwargs:
                layers.append(nn.MaxPool2d(kernel_size=kwargs['maxpool_kernels'][i], stride=kwargs['maxpool_strides'][i]))
        in_c = out_c
    return nn.Sequential(*layers)

def deconv_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act, batchnorm, **kwargs):
    """
    building blocks for a deconvnet
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=kwargs['leak_slope'], inplace=True)
    else:
        act = getattr(nn, activation)()
    assert len(channels) == len(kernels), "length of channels: %s,  length of kernels: %s" % (len(channels), len(kernels))
    assert len(channels) == len(strides), "length of channels: %s,  length of strides: %s" % (len(channels), len(strides))
    assert len(channels) == len(paddings), "length of channels: %s,  length of kernels: %s" % (len(channels), len(paddings))
    layers = []
    in_c = input_channels
    for i, out_c in enumerate(channels):
        layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        if (i < (len(channels)-1)) or last_act:
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_c)) 
            layers.append(act)
        in_c = out_c
    return nn.Sequential(*layers)

def mlp_block(input_dim, hidden_dims, activation, **kwargs):
    """
    building blocks for a mlp
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=kwargs['leak_slope'], inplace=True)
    else:
        act = getattr(nn, activation)()
    layers = []
    in_dim = input_dim
    for i, out_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(act)
        in_dim = out_dim
    return nn.Sequential(*layers)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape) 

def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)

    h = (h_w[0] + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1

    return h, w

def deconv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of deconvolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)
    h = (h_w[0] - 1) * stride[0] - 2 * padding[0]  + (dilation * (kernel_size[0] - 1)) + 1
    w = (h_w[1] - 1) * stride[1] - 2 * padding[1]  + (dilation * (kernel_size[1] - 1)) + 1

    return h, w

def cnn_output_shape(h, w, kernels, strides, paddings):
    h_w = (h, w)
    for i, kernel in enumerate(kernels):
        h_w = conv_output_shape(h_w, kernels[i], strides[i], paddings[i])
    return h_w

def dcnn_output_shape(h, w, kernels, strides, paddings):
    h_w = (h, w)
    for i, kernel in enumerate(kernels):
        h_w = deconv_output_shape(h_w, kernels[i], strides[i], paddings[i])
    return h_w


# ### 1.1.5 GraphEBM:

# In[ ]:


class GraphEBM(nn.Module):
    """
    E.g. GraphEBM(models_dict), where
        models = {
            "re_0": model_re.set_c(OPERATORS["SameShape"].get_node_repr()[None]),
            "re_1": model_re.set_c(OPERATORS["IsInside"].get_node_repr()[None]),
            "obj_0": model_concept.set_c(CONCEPTS["Rect"].get_node_repr()[None]),
        }
        assign_dict = {
            "obj_0": {2},
            "re_0": {(0, 1)},
            "re_1": {(1, 2)},
        }
    """
    def __init__(
        self,
        models,
        assign_dict,
        mask_arity,
    ):
        super().__init__()
        self.models = models
        self.assign_dict = assign_dict
        self.mask_arity = mask_arity

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def init_assign_dict_param(self):
        self.assign_dict_param = nn.ParameterDict()
        for key, List in self.assign_dict.items():
            self.assign_dict_param["{}^{}".format(key, List)] = nn.Parameter(1+torch.randn(1).to(self.device)*0.01)

    def del_assign_dict_param(self):
        delattr(self, "assign_dict_param")

    def forward(self, input, mask, c_repr=None):
        """
        Args:
            input:  x, [B, C, H, W]
            mask:   a list of masks, each with shape [B, 1, H, W]
            c_repr: embedding, [B, REPR_DIM]
        """
        energy = 0
        self.info = {}
        for key, mask_ids in self.assign_dict.items():
            assert isinstance(mask_ids, set)
            for mask_ids_setele in mask_ids:
                if not isinstance(mask_ids_setele, tuple):
                    mask_ids_setele = (mask_ids_setele,)
                mask_ele = tuple(mask[id] for id in mask_ids_setele)
                if len(mask_ele) == 1:
                    input_ele = input
                else:
                    assert len(mask_ele) == 2
                    input_ele = (input, input)
                energy_ele = self.models[key](input_ele, mask_ele)
                if hasattr(self, "assign_dict_param"):
                    energy_ele = energy_ele * self.assign_dict_param["{}^{}".format(key, mask_ids)]
                energy = energy + energy_ele
                string = ",".join([str(ele) for ele in mask_ids_setele])
                string = "({})".format(string) if len(mask_ids_setele) > 1 else string
                self.info["{}^{}".format(key, string)] = to_np_array(energy_ele)
        return energy

    def infer_recur(
        self,
        model,
        input,
        mask,
        max_recur_depth=6,
    ):
        """Recursively infer the concept component of the mask."""
        mask_list = []
        for i in range(max_recur_depth):
            neg_mask_ensemble_sorted, neg_out_ensemble_sorted = model.ground(input, args=args_concept, mask=mask, ensemble_size=1)
            mask_c = tuple(mask_ele[:,0] for mask_ele in neg_mask_ensemble_sorted)
            mask_c = and_mask(mask_c, mask)
            mask = subtract_mask(mask, mask_c)
            mask_list.append(mask_c)
            if mask[0].sum() == 0:
                print("break at {}".format(i))
                break
        masks_c = Zip(*mask_list, function=torch.cat)
        return masks_c

    
    def classify(
        self,
        input,
        mask,
        concept_collection,
        CONCEPTS=None,
        OPERATORS=None,
    ):
        """
        Uses a corresponding model to classify each mask in self.assign_dict.
        """
        
        pred_dict = {}
        for key, mask_ids in self.assign_dict.items():
            assert isinstance(mask_ids, set)
            for mask_ids_setele in mask_ids:
                if not isinstance(mask_ids_setele, tuple):
                    mask_ids_setele = (mask_ids_setele,)
                mask_ele = tuple(mask[id] for id in mask_ids_setele)
                if len(mask_ele) == 1:
                    input_ele = input
                else:
                    assert len(mask_ele) == 2
                    input_ele = (input, input)
                pred_ele = self.models[key].classify(input_ele, mask_ele, concept_collection=concept_collection[key], CONCEPTS=CONCEPTS, OPERATORS=OPERATORS)
                string = ",".join([str(ele) for ele in mask_ids_setele])
                string = "({})".format(string) if len(mask_ids_setele) > 1 else string
                pred_dict["{}^{}".format(key, string)] = pred_ele
        return pred_dict

    def ground(
        self,
        input,
        args,
        mask=None,
        c_repr=None,
        z=None,
        ensemble_size=18,
        topk=-1,
        w_init_type="random",
        sample_step=150,
        isplot=2,
        ground_truth_mask: tuple = None,
        **kwargs
    ):
        """
        Given input (and optionally initial masks), find the best masks that
        minimize the total energy.

        Args:
            isplot: whether or not to plot. If 0, skips plotting. If 1,
                only plots individual discovered masks. If 2, plots learning curve as well.

            ground_truth_mask: the real (positive) mask. If given, ground()
                will plot the energy of the ground truth masks as well. Tuple of tensors
                with shape [1, C, H, W] and length mask_arity.
        """
        def init_neg_mask(input, init, ensemble_size):
            """Initialize negative mask"""
            device = input.device
            neg_mask = tuple(torch.rand(input.shape[0]*ensemble_size, 1, *input.shape[2:]).to(device) for k in range(self.mask_arity))
            if init == "input-mask":
                assert input.shape[1] == 10
                input_l = repeat_n(input, n_repeats=ensemble_size)
                neg_mask = tuple(neg_mask[k] * (input_l.argmax(1)[:, None] != 0) for k in range(self.mask_arity))
            for k in range(self.mask_arity):
                neg_mask[k].requires_grad = True
            return neg_mask

        def plot_discovered_mask_summary(num_examples: int, neg_mask_ensemble_sorted: torch.Tensor, should_quantize: bool):
            plt.figure(figsize=(18,3))
            for batch_idx in range(len(neg_mask_ensemble_sorted[0])):
                for ex in range(num_examples):
                    for mask_idx in range(len(neg_mask_ensemble_sorted)):
                        ax = plt.subplot(1, num_examples, ex + 1)
                        # Pull single-channel (0)
                        mask = neg_mask_ensemble_sorted[mask_idx][batch_idx][ex][0].cpu()
                        image = np.zeros((*mask.shape, 4)) # (H, W, C, alpha)
                        color = np.asarray(matplotlib.colors.to_rgb(COLOR_LIST[mask_idx]))
                        for h in range(mask.shape[0]):
                            for w in range(mask.shape[1]):
                                opacity = torch.round(mask[h][w]) if should_quantize else mask[h][w]
                                pixel = opacity * np.asarray((*color, 0.5)) # add alpha channel
                                image[h][w] = pixel
                        plt.imshow(image)
                        ax.set_title("E: {:.5f}\n".format(neg_out_ensemble_sorted[batch_idx][ex]))
            plt.show()

        assert (not isinstance(input, tuple)) and (not isinstance(input, list)) and len(input.shape) == 4
        device = input.device
        # Update args:
        args = deepcopy(args)
        for key, value in kwargs.items():
            setattr(args, key, value)
        args.sample_step = sample_step
        z_mode = self.models[next(iter(self.models))].z_mode
        args.ebm_target = "mask" if z_mode == "None" else "mask+z"

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

        print("Inferred masks:")
        # Perform SGLD:
        if mask is None:
            neg_mask = init_neg_mask(input, init=w_init_type, ensemble_size=ensemble_size)
        else:
            neg_mask = mask
        (img_ensemble, neg_mask_ensemble, z_ensemble), neg_out_list_ensemble = neg_mask_sgd_ensemble(
            self, input, neg_mask, c_repr, z=z, args=args,
            ensemble_size=ensemble_size,
        )
        neg_out_ensemble = neg_out_list_ensemble[-1]  # [time_steps, ensemble_size, B]
        # Sort the obtained results by energy for each example:
        neg_out_ensemble = torch.FloatTensor(neg_out_ensemble).transpose(0,1)
        neg_out_argsort = neg_out_ensemble.argsort(1)  # [B, ensemble_size]
        batch_size = neg_out_argsort.shape[0]
        neg_out_ensemble_sorted = torch.stack([neg_out_ensemble[i][neg_out_argsort[i]] for i in range(len(neg_mask_ensemble[0]))])  # [B, ensemble_size]

        if img_ensemble is not None:
            if args.is_image_tuple:
                img_ensemble = tuple(img_ensemble[k].transpose(0,1) for k in range(len(img)))  # Each element [B, ensemble_size, C, H, W]
                img_ensemble_sorted = []
                for k in range(len(img)):
                    img_ensemble_sorted.append(torch.stack([img_ensemble[k][i][neg_out_argsort[i]] for i in range(batch_size)]))
                img_ensemble_sorted = tuple(img_ensemble_sorted)  # each element: [B, ensemble_size, C, H, W] sorted along dim=1 according to neg_out
            else:
                img_ensemble = img_ensemble.transpose(0,1)  # [B, ensemble_size, C, H, W]
                img_ensemble_sorted = torch.stack([img_ensemble[i][neg_out_argsort[i]] for i in range(batch_size)])
        else:
            img_ensemble_sorted = None

        if neg_mask_ensemble is not None:
            neg_mask_ensemble = tuple(neg_mask_ensemble[k].transpose(0,1) for k in range(self.mask_arity))  # Each element [B, ensemble_size, C, H, W]
            neg_mask_ensemble_sorted = []
            for k in range(self.mask_arity):
                neg_mask_ensemble_sorted.append(torch.stack([neg_mask_ensemble[k][i][neg_out_argsort[i]] for i in range(len(neg_mask_ensemble[0]))]))
            # Shape: tuple of length mask_arity
            # where each element: [B, ensemble_size, C, H, W] sorted along dim=1 according to neg_out
            # (aka, lowest energy configuration first)
            neg_mask_ensemble_sorted = tuple(neg_mask_ensemble_sorted)  # each element: [B, ensemble_size, C, H, W] sorted along dim=1 according to neg_out
        else:
            neg_mask_ensemble_sorted = None

        if z_ensemble is not None:
            z_ensemble = tuple(z_ensemble[k].transpose(0,1) for k in range(len(z_ensemble)))  # Each element [B, ensemble_size, Z]
            z_ensemble_sorted = []
            for k in range(len(z_ensemble)):
                z_ensemble_sorted.append(torch.stack([z_ensemble[k][i][neg_out_argsort[i]] for i in range(batch_size)]))
            z_ensemble_sorted = tuple(z_ensemble_sorted)  # each element: [B, ensemble_size, Z] sorted along dim=1 according to neg_out
        else:
            z_ensemble_sorted = None

        # Obtain each individual energy for component models:
        info = {}
        neg_out_argsort = to_np_array(neg_out_argsort)
        for key, value in self.info.items():
            value_reshape = value.reshape(ensemble_size, -1).T  # [B, ensemble_size]
            info[key] = []
            for i in range(len(value_reshape)):
                info[key].append(value_reshape[i][neg_out_argsort[i]])
            info[key] = np.stack(info[key])

        NUM_PREV_EXAMPLES = 6

        if isplot >= 2:
            # Plot SGLD learning curve:
            plt.figure(figsize=(12,6))
            for i in range(min(neg_out_list_ensemble.shape[-1], 6)):  # neg_out_list_ensemble: [sample_step, ensemble_size, B]
                print("Example {}".format(i))
                for k in range(6):
                    plt.plot(neg_out_list_ensemble[:,neg_out_argsort[i][k],i], c=COLOR_LIST[k], label="id_{}".format(k), alpha=0.4)
            plt.legend()
            plt.show()
        if isplot >= 1:
            # Show original input image for reference
            print("Original inputs:")
            visualize_matrices(input.argmax(1).repeat_interleave(NUM_PREV_EXAMPLES, 0))

            # Plot a summary plot, superimposing different masks such that each mask has a different color
            print(f"Top-{NUM_PREV_EXAMPLES} lowest-energy mask sets, all plotted together")
            print(f"Key parameters: SGLD_mutual_exclusive_coef={str(args.SGLD_mutual_exclusive_coef)}",
                    f"SGLD_object_exceed_coef={str(args.SGLD_object_exceed_coef)}")
            plot_discovered_mask_summary(NUM_PREV_EXAMPLES, neg_mask_ensemble_sorted, should_quantize=False)

            # Plot the same plot, but this time quantized so there are no color gradations
            print("Quantized plot:")
            plot_discovered_mask_summary(NUM_PREV_EXAMPLES, neg_mask_ensemble_sorted, should_quantize=True)

            # For each batch element
            for i in range(len(neg_mask_ensemble_sorted[0])):
                print("Example {}".format(i))
                # Loop through each mask (show them horizontally)
                for k in range(len(neg_mask_ensemble_sorted)):
                    visualize_matrices(
                        torch.round(neg_mask_ensemble_sorted[k][i][:NUM_PREV_EXAMPLES].squeeze(1)), images_per_row=NUM_PREV_EXAMPLES,
                        subtitles=["E: {:.5f}\n".format(neg_out_ensemble_sorted[i][j]) + "\n".join(["{}: {:.5f}".format(key, info[key][i][j]) for key in info]) for j in range(NUM_PREV_EXAMPLES)] if k == 0 else None
                    )
        return (img_ensemble_sorted, neg_mask_ensemble_sorted, z_ensemble_sorted), neg_out_ensemble_sorted, info

    @property
    def model_dict(self):
        model_dict = {"type": self.__class__.__name__}
        model_dict["models"] = {key: model.model_dict for key, model in self.models.items()}
        model_dict["assign_dict"] = self.assign_dict
        model_dict["mask_arity"] = self.mask_arity
        return model_dict



class SumEBM(nn.Module):
    """
    SumEBM((model1, c_repr1, "Line"), (model2, c_repr2, "Vertical"))
    """
    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.modes = [model.mode for model in self.models]
        self.is_two_branch = True if "relation" in self.modes or "operator" in self.modes else False

    def forward(self, input, mask, c_repr):
        energy_list = []
        for model in self.models:
            if self.is_two_branch:
                if model.mode in ["relation", "operator"]:
                    energy = model(input, mask, model.c_repr)
                elif model.mode == "concept":
                    energy = model(input[0], (mask[0],), model.c_repr) + model(input[1], (mask[1],), model.c_repr)
                else:
                    raise Exception("The model's mode must be one of 'concept', 'relation' or 'operator'.")
            else:
                energy = model(input, mask, model.c_repr)
            energy_list.append(energy)
        energy = torch.cat(energy_list, -1).sum(-1, keepdims=True)
        return energy

    def __add__(self, model):
        if model.__class__.__name__ == "ConceptEBM":
            self.models.append(model)
            self.modes.append(model.mode)
        elif model.__class__.__name__ == "SumEBM":
            self.models = self.models + model.models
            self.modes += model.modes
        else:
            raise Exception("model is not valid!")
        self.is_two_branch = True if "relation" in self.modes or "operator" in self.modes else False
        return self

    @property
    def model_dict(self):
        model_dict = {"type": self.__class__.__name__}
        model_dict["models"] = [model.model_dict for model in self.models]
        return model_dict


def and_mask(mask, mask_ref):
    """Perform And operation that only preserve the non-zero part of mask_ref onto mask."""
    mask_and = []
    for mask1_ele, mask2_ele in zip(mask, mask_ref):
        mask_and_ele = (mask1_ele * mask2_ele.round().bool()).float()
        mask_and.append(mask_and_ele)
    return tuple(mask_and)

def subtract_mask(mask1, mask2):
    """Subtract mask1 by mask2 as Boolean."""
    mask_sub = []
    for mask1_ele, mask2_ele in zip(mask1, mask2):
        mask_sub_ele = (mask1_ele.round().bool() & ~mask2_ele.round().bool()).float()
        mask_sub.append(mask_sub_ele)
    return tuple(mask_sub)


# ## 1.2 Modules:

# ### 1.2.1 SpectralNorm:

# In[ ]:


class SpectralNorm(object):
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module


class WSConv2d(nn.Conv2d):
    """Conv layer with the output normalized."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean((1,2,3), keepdims=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# ### 1.2.2 ResBlock:

# In[ ]:


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_class=None, downsample=False, is_spec_norm=True):
        super().__init__()

        if is_spec_norm:
            self.conv1 = spectral_norm(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    3,
                    padding=1,
                    bias=False if n_class is not None else True,
                )
            )
            self.conv2 = spectral_norm(
                nn.Conv2d(
                    out_channel,
                    out_channel,
                    3,
                    padding=1,
                    bias=False if n_class is not None else True,
                ), std=1e-10, bound=True
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            )
            self.conv2 = nn.Conv2d(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            )

        self.class_embed = None

        if n_class is not None:
            class_embed = nn.Embedding(n_class, out_channel * 2 * 2)
            class_embed.weight.data[:, : out_channel * 2] = 1
            class_embed.weight.data[:, out_channel * 2 :] = 0

            self.class_embed = class_embed

        self.skip = None

        if in_channel != out_channel or downsample:
            if is_spec_norm:
                self.skip = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channel, out_channel, 1, bias=False))
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 1, bias=False)
                )

        self.downsample = downsample

    def forward(self, input, class_id=None):
        out = input

        out = self.conv1(out)

        if self.class_embed is not None:
            embed = self.class_embed(class_id).view(input.shape[0], -1, 1, 1)
            weight1, weight2, bias1, bias2 = embed.chunk(4, 1)
            out = weight1 * out + bias1

        out = F.leaky_relu(out, negative_slope=0.2)

        out = self.conv2(out)

        if self.class_embed is not None:
            out = weight2 * out + bias2

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = F.leaky_relu(out, negative_slope=0.2)

        return out


# ### 1.2.3 CResBlock:

# In[ ]:


class CResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        repr_dim=None,
        downsample=False,
        is_spec_norm=True,
        c_repr_mode="l1",
        c_repr_base=2,
        z_mode="None",
        z_dim=4,
        img_dims=2,
        act_name="leakyrelu0.2",
        normalization_type="None",
        dropout=0,
        is_res=True,
        downsample_mode="None",
    ):
        super().__init__()

        in_channel_combined = in_channel
        out_channel_combined = out_channel
        if c_repr_mode.startswith("c"):
            in_channel_combined += c_repr_base
            out_channel_combined += c_repr_base
        if z_mode.startswith("c"):
            in_channel_combined += z_dim
            out_channel_combined += z_dim
        self.img_dims = img_dims
        if img_dims == 2:
            kernel_size = 3
            padding = 1
            self.pool_kernel_size = 2
        elif img_dims == 1:
            kernel_size = (3, 1)
            padding = (1, 0)
            self.pool_kernel_size = (2, 1)
        else:
            raise

        self.act1 = get_activation(act_name)
        self.act2 = get_activation(act_name)
        self.bn1 = get_normalization(normalization_type, in_channels=out_channel)
        self.bn2 = get_normalization(normalization_type, in_channels=out_channel)
        self.dropout = dropout
        self.is_res = is_res
        self.downsample = downsample
        self.downsample_mode = downsample_mode
        if self.dropout != 0:
            self.dropout_fn = nn.Dropout(dropout)

        if is_spec_norm in [True, "True"]:
            self.conv1 = spectral_norm(
                nn.Conv2d(
                    in_channel_combined,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            self.conv2 = spectral_norm(
                nn.Conv2d(
                    out_channel_combined,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    bias=False,
                ), std=1e-10, bound=True
            )
        elif is_spec_norm in [False, "False"]:
            self.conv1 = nn.Conv2d(
                in_channel_combined,
                out_channel,
                kernel_size,
                padding=padding,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                out_channel_combined,
                out_channel,
                kernel_size,
                padding=padding,
                bias=False,
            )
        elif is_spec_norm == "ws":
            self.conv1 = WSConv2d(
                in_channel_combined,
                out_channel,
                kernel_size,
                padding=padding,
                bias=False,
            )
            self.conv2 = WSConv2d(
                out_channel_combined,
                out_channel,
                kernel_size,
                padding=padding,
                bias=False,
            )
        else:
            raise

        # Constructing self.class_embed:
        self.c_repr_mode = c_repr_mode
        self.c_repr_base = c_repr_base
        if c_repr_mode.startswith("l1"):
            self.class_embed = nn.Linear(repr_dim, out_channel * 4)
        elif c_repr_mode.startswith("l2"):
            self.class_embed = nn.Sequential(nn.Linear(repr_dim, repr_dim * c_repr_base*2),
                                             nn.LeakyReLU(negative_slope=0.2),
                                             nn.Linear(repr_dim * c_repr_base*2, out_channel * 4),
                                            )
        elif c_repr_mode.startswith("l3"):
            self.class_embed = nn.Sequential(nn.Linear(repr_dim, repr_dim * c_repr_base*2),
                                             nn.LeakyReLU(negative_slope=0.2),
                                             nn.Linear(repr_dim * c_repr_base*2, repr_dim * c_repr_base*2),
                                             nn.LeakyReLU(negative_slope=0.2),
                                             nn.Linear(repr_dim * c_repr_base*2, out_channel * 4),
                                            )
        elif c_repr_mode.startswith("c1"):
            self.class_embed = nn.Linear(repr_dim, c_repr_base*2)
        elif c_repr_mode.startswith("c2"):
            self.class_embed = nn.Sequential(nn.Linear(repr_dim, repr_dim * c_repr_base*2),
                                             nn.LeakyReLU(negative_slope=0.2),
                                             nn.Linear(repr_dim * c_repr_base*2, c_repr_base*2),
                                            )
        elif c_repr_mode.startswith("c3"):
            self.class_embed = nn.Sequential(nn.Linear(repr_dim, repr_dim * c_repr_base*2),
                                             nn.LeakyReLU(negative_slope=0.2),
                                             nn.Linear(repr_dim * c_repr_base*2, repr_dim * c_repr_base*2),
                                             nn.LeakyReLU(negative_slope=0.2),
                                             nn.Linear(repr_dim * c_repr_base*2, c_repr_base*2),
                                            )
        elif c_repr_mode == "None":
            pass
        else:
            raise

        # Constructing self.z_embed:
        self.z_mode = z_mode
        self.z_dim = z_dim
        if z_mode.startswith("c0"):
            def repeat_dim(z):
                return torch.cat([z,z], 1)
            self.z_embed = repeat_dim
        elif z_mode.startswith("c1"):
            self.z_embed = nn.Linear(z_dim, z_dim*2)
        elif z_mode.startswith("c2"):
            self.z_embed = nn.Sequential(nn.Linear(z_dim, z_dim*4),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Linear(z_dim*4, z_dim*2),
                                        )
        elif z_mode.startswith("c3"):
            self.z_embed = nn.Sequential(nn.Linear(z_dim, z_dim*4),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Linear(z_dim*4, z_dim*4),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Linear(z_dim*4, z_dim*2),
                                        )
        elif z_mode == "None":
            pass
        else:
            raise

        self.skip = None

        if in_channel != out_channel or downsample:
            if is_spec_norm in [True, "True"]:
                self.skip = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channel, out_channel, 1, bias=False))
                )
            elif is_spec_norm in [False, "False"]:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 1, bias=False)
                )
            elif is_spec_norm == "ws":
                self.skip = nn.Sequential(
                    WSConv2d(in_channel, out_channel, 1, bias=False)
                )

        if self.downsample_mode != "None":
            assert self.downsample_mode in ["conv", "conv+rescale"]
            if self.downsample_mode == "conv":
                self.conv_downsample = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)
            elif self.downsample_mode == "conv+rescale":
                self.conv_downsample = nn.Conv2d(out_channel, out_channel*2, kernel_size=kernel_size, padding=padding)

    def forward(self, input, c_repr=None, z=None):
        out = input
        if c_repr is not None:
            if "softmax" in self.c_repr_mode:
                c_repr = F.softmax(c_repr, dim=-1)
            c_embed = self.class_embed(c_repr).view(input.shape[0], -1, 1, 1)  # [B, c_dim*4, 1, 1]
        if z is not None:
            assert (isinstance(z, tuple) or isinstance(z, list)) and len(z) == 1
            z = z[0]
            z_embed = self.z_embed(z).view(input.shape[0], -1, 1, 1)

        if self.c_repr_mode == "None":
            assert c_repr is None
            if self.z_mode == "None":
                out = self.conv1(out)
                out = self.bn1(out)
                out = self.act1(out)
                if hasattr(self, "dropout") and self.dropout != 0:
                    out = self.dropout_fn(out)
                out = self.conv2(out)
                out = self.bn2(out)
            elif self.z_mode.startswith("c"):
                z_embed_1, z_embed_2 = z_embed.chunk(2, 1)
                out_shape = out.shape[2:]
                out = self.conv1(torch.cat([out, z_embed_1.expand(*z_embed_1.shape[:2], *out_shape)], 1))
                out = self.bn1(out)
                out = self.act1(out)
                if hasattr(self, "dropout") and self.dropout != 0:
                    out = self.dropout_fn(out)
                out = self.conv2(torch.cat([out, z_embed_2.expand(*z_embed_2.shape[:2], *out_shape)], 1))
                out = self.bn2(out)
            else:
                raise
        elif self.c_repr_mode.startswith("l"):
            if self.z_mode == "None":
                weight1, weight2, bias1, bias2 = c_embed.chunk(4, 1) # [B, out_channel, 1, 1]
                out = self.conv1(out)    # [B, out_channel, H, W]
                out = self.bn1(out)
                out = weight1 * out + bias1
                out = self.act1(out)
                if hasattr(self, "dropout") and self.dropout != 0:
                    out = self.dropout_fn(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = weight2 * out + bias2
            elif self.z_mode.startswith("c"):
                z_embed_1, z_embed_2 = z_embed.chunk(2, 1)
                out_shape = out.shape[2:]
                weight1, weight2, bias1, bias2 = c_embed.chunk(4, 1)
                out = self.conv1(torch.cat([out, z_embed_1.expand(*z_embed_1.shape[:2], *out_shape)], 1))
                out = self.bn1(out)
                out = weight1 * out + bias1
                out = self.act1(out)
                if hasattr(self, "dropout") and self.dropout != 0:
                    out = self.dropout_fn(out)
                out = self.conv2(torch.cat([out, z_embed_2.expand(*z_embed_2.shape[:2], *out_shape)], 1))
                out = self.bn2(out)
                out = weight2 * out + bias2
            else:
                raise
        elif self.c_repr_mode.startswith("c"):
            if self.z_mode == "None":
                c_embed_1, c_embed_2 = c_embed.chunk(2, 1)
                out_shape = out.shape[2:]
                out = self.conv1(torch.cat([out, c_embed_1.expand(*c_embed_1.shape[:2], *out_shape)], 1))
                out = self.bn1(out)
                out = self.act1(out)
                if hasattr(self, "dropout") and self.dropout != 0:
                    out = self.dropout_fn(out)
                out = self.conv2(torch.cat([out, c_embed_2.expand(*c_embed_2.shape[:2], *out_shape)], 1))
                out = self.bn2(out)
            elif self.z_mode.startswith("c"):
                c_embed_1, c_embed_2 = c_embed.chunk(2, 1)
                z_embed_1, z_embed_2 = z_embed.chunk(2, 1)
                out_shape = out.shape[2:]
                out = self.conv1(torch.cat([out, z_embed_1.expand(*z_embed_1.shape[:2], *out_shape), c_embed_1.expand(*c_embed_1.shape[:2], *out_shape)], 1))
                out = self.bn1(out)
                out = self.act1(out)
                if hasattr(self, "dropout") and self.dropout != 0:
                    out = self.dropout_fn(out)
                out = self.conv2(torch.cat([out, z_embed_2.expand(*z_embed_2.shape[:2], *out_shape), c_embed_2.expand(*c_embed_2.shape[:2], *out_shape)], 1))
                out = self.bn2(out)
        else:
            raise

        if hasattr(self, "is_res") and self.is_res:
            if self.skip is not None:
                skip = self.skip(input)
            else:
                skip = input
            out = out + skip

        if self.downsample:
            if not hasattr(self, "downsample_mode") or self.downsample_mode == "None":
                out = F.avg_pool2d(out, self.pool_kernel_size)
            elif self.downsample_mode in ["conv", "conv+rescale"]:
                out = self.conv_downsample(out)
                out = F.avg_pool2d(out, self.pool_kernel_size)
            else:
                raise

        out = self.act2(out)

        return out


# ### 1.2.4 Self-attention:

# In[ ]:


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, act_name):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = get_activation(act_name)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B x N x N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B x (N) x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*H*W)
        energy = torch.bmm(proj_query, proj_key) # B x N x N
        attention = self.softmax(energy) # B x (N) x (N), for each of the N queries, perform softmax on the N keys
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1)) # value: BxCxN,  attention_perm: BxNxN: attention on the values.
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out, attention


# ### 1.2.5 PositionEmbedding:

# In[ ]:


# Adapted from https://github.com/vadimkantorov/yet_another_pytorch_slot_attention/blob/master/models.py
class PositionEmbeddingImplicit(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(4, hidden_dim)

    def forward(self, x):
        spatial_shape = x.shape[-3:-1]
        grid = torch.stack(torch.meshgrid(*[torch.linspace(0., 1., r, device = x.device) for r in spatial_shape]), dim = -1)
        grid = torch.cat([grid, 1 - grid], dim = -1)
        return x + self.dense(grid)

class PositionEmbeddingSine(nn.Module):
    # https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        not_mask = torch.ones_like(x)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return x + pos


class PositionEmbeddingLearned(nn.Module):
    # https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        #TODO: assert that x.shape matches the passed row_embed, col_embed
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return x + pos


# ### 1.2.6 Downsample:

# In[ ]:


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


# In[ ]:


# if __name__ == "__main__":
#     # BabyARC-relation dataset, 3D:
#     from reasoning.experiments.concept_energy import get_dataset, ConceptDataset, ConceptDataset3D
#     relation_args = init_args({
#         "dataset": "y-Parallel+VerticalMid+VerticalEdge",
#         "seed": 2,
#         "n_examples": 40,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "color_map_3d": "same",
#         "add_thick_surf": (0, 0.5),
#         "add_thick_depth": (0, 0.5),
#         "max_n_distractors": 2,
#         "seed_3d": 42,
#         "num_processes_3d": 10,
#         "image_size_3d": (256,256),
#     })
#     relation_dataset, args = get_dataset(relation_args, is_rewrite=False, is_load=True)
#     relation_dataset.draw(range(40))
#     tensor = relation_dataset[0][0][None]
#     visualize_matrices([tensor[0]], use_color_dict=False)
#     dd = Downsample(channels=3)
#     tensor_d = dd(tensor)
#     visualize_matrices([tensor_d[0]], use_color_dict=False)


# ## 1.3 EBM Optimization:

# ### 1.3.1 SGLD ensemble:

# In[ ]:


def neg_mask_sgd_ensemble(
    model,
    img,
    neg_mask,
    c_repr,
    z=None,
    zgnn=None,
    wtarget=None,
    args=None,
    ensemble_size=None,
    out_mode="all",
    mask_info=None,
    record_interval=-1,
    return_history=False,
    is_grad=False,
    is_return_E=False,
    batch_shape=None,
):
    """
    Perform an ensemble to discover a mask - run the model {ensemble_size}
    times on the same image, and either return all the masks (out_mode) or
    the one with the "minimum" energy.

    The size of neg_mask and noise must be ensemble_size time that of
    img and c_repr, but pos_img should not be duplicated manually.

    Args:
        record_interval: if set, indicates the intermediate mask should
            be recorded every `record_interval` steps. Implies you should enable
            `return_mask_history`.
        return_history: whether the intermediate img/mask/z should be
            returned at every step. If enabled, a third tuple element will be
            returned.
        is_grad: if True, will use neg_mask_sgd_with_kl. Otherwise use neg_mask_sgd.
        batch_shape: if not None, will be (B_task, B_example).
        mask_info: dictionary for various constraint for mask. The key includes:
            "mask_exclude": if its value is not None, will have energy for the overlapping between mask_exclude and neg_mask, 
                with coefficient of SGLD_mutual_exclusive_coef.
    """
    # repeat ensemble_size time along the batch dimension:
    img_l, c_repr_l = repeat_n(img, c_repr, n_repeats=ensemble_size)
    if mask_info is not None and "mask_exclude" in mask_info:
        mask_info = deepcopy(mask_info)
        mask_info["mask_exclude"] = repeat_n(mask_info["mask_exclude"], n_repeats=ensemble_size)

    # Find neg_mask for the batch:
    if is_grad:
        _, (img, neg_mask, c_repr, z, zgnn, wtarget), info = neg_mask_sgd_with_kl(
            model,
            img=img_l, neg_mask=neg_mask, c_repr=c_repr_l, z=z, zgnn=zgnn, wtarget=wtarget,
            args=args,
            mask_info=mask_info,
            is_return_E=is_return_E,
            batch_shape=batch_shape,
            record_interval=record_interval,
        )
    else:
        (img, neg_mask, c_repr, z, zgnn, wtarget), info = neg_mask_sgd(
            model,
            img=img_l, neg_mask=neg_mask, c_repr=c_repr_l, z=z, zgnn=zgnn, wtarget=wtarget,
            args=args,
            mask_info=mask_info,
            is_return_E=is_return_E,
            batch_shape=batch_shape,
            record_interval=record_interval,
        )

    for key in info:
        if key.endswith("_list") and key not in ["img_list", "neg_mask_list", "c_repr_list", "z_list", "zgnn_list", "wtarget_list"]:
            info[key] = info[key].reshape(len(info[key]), ensemble_size, -1)  # [sample_step, ensemble_size, batch_size]
        else:
            if isinstance(info[key], tuple) or isinstance(info[key], list):
                if len(info[key]) > 0:
                    shape_rest = info[key][0].shape[2:]
                    info[key] = tuple(info[key][k].reshape(len(info[key][k]), ensemble_size, -1, *shape_rest) for k in range(len(info[key])))
            else:
                shape_rest = info[key].shape[2:]
                info[key] = info[key].reshape(len(info[key]), ensemble_size, -1, *shape_rest)

    neg_out_list_reshape = info.pop("neg_out_list")

    if "image" in args.ebm_target.split("+"):
        if args.is_image_tuple:
            img_reshape = tuple(img[k].reshape(
                ensemble_size, -1, *img[k].shape[1:]) for k in range(len(img)))  # [ensemble_size, batch_size, C, H, W]
        else:
            img_reshape = img.reshape(ensemble_size, -1, *img.shape[1:])  # [ensemble_size, batch_size, C, H, W]
    else:
        img_reshape = None
    if "mask" in args.ebm_target.split("+"):
        neg_mask_reshape = tuple(neg_mask[k].reshape(
            ensemble_size, -1, *neg_mask[k].shape[1:]) for k in range(len(neg_mask)))  # [ensemble_size, batch_size, C, H, W]
    else:
        neg_mask_reshape = None
    if "z" in args.ebm_target.split("+"):
        z_reshape = tuple(z[k].reshape(
            ensemble_size, -1, *z[k].shape[1:]) if z[k] is not None else None for k in range(len(z)))  # Each [ensemble_size, batch_size, Z]
    else:
        z_reshape = None
    if "zgnn" in args.ebm_target.split("+"):
        assert ensemble_size == 1
        zgnn_reshape = tuple(zgnn[k].reshape(
            ensemble_size, -1, *zgnn[k].shape[1:]) if zgnn[k] is not None else None for k in range(len(zgnn)))  # Each [ensemble_size, batch_size, Z]
    else:
        zgnn_reshape = None
    if "wtarget" in args.ebm_target.split("+"):
        assert ensemble_size == 1
        wtarget_reshape = wtarget.reshape(ensemble_size, -1, *wtarget.shape[1:])  # [ensemble_size, batch_size, C, H, W]
    else:
        wtarget_reshape = None

    if out_mode == "min":
        idx_argmin = neg_out_list_reshape[-1].argmin(0)  # [B,] each number has value up to ensemble_size-1
        if "image" in args.ebm_target.split("+"):
            if args.is_image_tuple:
                img_reshape = tuple(gather_broadcast(img_reshape[k].transpose(
                    0, 1), 1, idx_argmin) for k in range(len(img)))
            else:
                img_reshape = gather_broadcast(img_reshape.transpose(
                    0, 1), 1, idx_argmin)
        if "mask" in args.ebm_target.split("+"):
            neg_mask_reshape = tuple(gather_broadcast(neg_mask_reshape[k].transpose(
                0, 1), 1, idx_argmin) for k in range(len(neg_mask)))
        if "z" in args.ebm_target.split("+"):
            z_reshape = tuple(gather_broadcast(z_reshape[k].transpose(
                0, 1), 1, idx_argmin) if z[k] is not None else None for k in range(len(z)))
        if "zgnn" in args.ebm_target.split("+"):
            zgnn_reshape = tuple(gather_broadcast(zgnn_reshape[k].transpose(
                0, 1), 1, idx_argmin) if zgnn[k] is not None else None for k in range(len(zgnn)))
        if "wtarget" in args.ebm_target.split("+"):
            wtarget_reshape = gather_broadcast(wtarget_reshape.transpose(
                0, 1), 1, idx_argmin)
    elif out_mode == "all":
        pass
    elif out_mode == "all-sorted":
        idx_argsort = torch.LongTensor(neg_out_list_reshape[-1].argsort(0))  # [ensemble_size, B] each number has value up to ensemble_size-1
        # neg_out_list_reshape (shape of [sample_step, ensemble_size, B]):
        index = torch.LongTensor(idx_argsort[None]).expand(neg_out_list_reshape.shape).numpy()
        neg_out_list_reshape = np.take_along_axis(neg_out_list_reshape, indices=index, axis=1)
        # Other results:
        if "image" in args.ebm_target.split("+"):
            if args.is_image_tuple:
                index = idx_argsort[:,:,None,None,None].to(img_reshape[0].device).expand_as(img_reshape[0])  # [ensemble_size, B, ...]
                img_reshape = tuple(torch.gather(img_reshape[k], dim=0, index=index) for k in range(len(img)))
            else:
                index = idx_argsort[:,:,None,None,None].to(img_reshape.device).expand_as(img_reshape)
                img_reshape = torch.gather(img_reshape, dim=0, index=index)
        if "mask" in args.ebm_target.split("+"):
            index = idx_argsort[:,:,None,None,None].to(neg_mask_reshape[0].device).expand_as(neg_mask_reshape[0])  # [ensemble_size, B, ...]
            neg_mask_reshape = tuple(torch.gather(neg_mask_reshape[k], dim=0, index=index) for k in range(len(neg_mask_reshape)))
        if "z" in args.ebm_target.split("+"):
            index = idx_argsort[:,:,None].to(z_reshape[0].device).expand_as(z_reshape[0])
            z_reshape = tuple(torch.gather(z_reshape[k], dim=0, index=index) if z[k] is not None else None for k in range(len(z)))
        if "zgnn" in args.ebm_target.split("+"):
            index = idx_argsort[:,:,None].to(zgnn_reshape[0].device).expand_as(zgnn_reshape[0])
            zgnn_reshape = tuple(torch.gather(zgnn_reshape[k], dim=0, index=index) if zgnn[k] is not None else None for k in range(len(zgnn)))
        if "wtarget" in args.ebm_target.split("+"):
            index = idx_argsort[:,:,None,None,None].to(wtarget_reshape.device).expand_as(wtarget_reshape)
            wtarget_reshape = torch.gather(wtarget_reshape, dim=0, index=index)
        # model.info:
        if hasattr(model, "info"):
            for key, value in model.info.items():
                index = idx_argsort.numpy()  # [ensemble_size, B]
                model.info[key] = np.take_along_axis(value, indices=index, axis=0)
        # info:
        for key, value in info.items():
            if key.endswith("_list"):
                if key not in ["img_list", "neg_mask_list", "c_repr_list", "z_list", "zgnn_list", "wtarget_list"]:
                    # The value has shape [sample_step, ensemble_size, B]:
                    index = torch.LongTensor(idx_argsort[None]).expand(value.shape).numpy()
                    info[key] = np.take_along_axis(value, indices=index, axis=1)
                elif len(value) > 0:
                    if not isinstance(value, tuple) and not isinstance(value, list):
                        index = torch.LongTensor(extend_dims(idx_argsort[None], n_dims=len(value.shape), loc="right")).expand(value.shape).numpy()
                        info[key] = np.take_along_axis(value, indices=index, axis=1)
                    else:
                        index = torch.LongTensor(extend_dims(idx_argsort[None], n_dims=len(value[0].shape), loc="right")).expand(value[0].shape).numpy()
                        info[key] = tuple(np.take_along_axis(value_ele, indices=index, axis=1) for value_ele in value)
    else:
        raise

    if return_history or is_return_E:
        return (img_reshape, neg_mask_reshape, z_reshape, zgnn_reshape, wtarget_reshape), neg_out_list_reshape, info
    else:
        return (img_reshape, neg_mask_reshape, z_reshape, zgnn_reshape, wtarget_reshape), neg_out_list_reshape, {}


# ### 1.3.2 SGLD:

# In[ ]:


def neg_mask_sgd(
    model,
    img=None,
    neg_mask=None,
    c_repr=None,
    z=None,
    zgnn=None,
    wtarget=None,
    args=None,
    mask_info=None,
    is_return_E=False,
    batch_shape=None,
    record_interval=-1,
):
    """Perform SGLD w.r.t. a subset of {img, neg_mask, c_repr, z} given the others."""
    requires_grad(model.parameters(), False)
    model.eval()
    neg_out_list = []
    if args.step_size_img == -1:
        args.step_size_img = args.step_size
    if args.step_size_z == -1:
        args.step_size_z = args.step_size
    if args.step_size_zgnn == -1:
        args.step_size_zgnn = args.step_size
    if args.step_size_wtarget == -1:
        args.step_size_wtarget = args.step_size

    # Initialize variables (a subset of {img, neg_mask, c_repr, z}):
    if img is not None:
        batch_size = img[0].shape[0] if args.is_image_tuple else img.shape[0]
        in_channels = img[0].shape[1] if args.is_image_tuple else img.shape[1]
        device = img[0].device if args.is_image_tuple else img.device
        img_shape = img[0].shape[2:] if args.is_image_tuple else img.shape[2:]
    else:
        batch_size = neg_mask[0].shape[0]
        in_channels = args.in_channels
        img_shape = args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size)
        device = neg_mask[0].device

    img_list = []
    if "image" in args.ebm_target.split("+"):
        img_value_min, img_value_max = args.image_value_range.split(",")
        img_value_min, img_value_max = eval(img_value_min), eval(img_value_max)
        img_value_span = img_value_max - img_value_min
        assert img_value_span >= 1
        if img is None:
            img = (
                torch.rand(batch_size, in_channels, *img_shape, device=device) * img_value_span + img_value_min, torch.rand(batch_size, in_channels, *img_shape, device=device) * img_value_span + img_value_min
            ) if args.is_image_tuple else torch.rand(batch_size, in_channels, *img_shape, device=device) * img_value_span + img_value_min
        else:
            if isinstance(img, tuple):
                img = tuple(torch.rand(batch_size, in_channels, *img_shape, device=device) * img_value_span + img_value_min if img_ele is None else img_ele for img_ele in img)
        if args.is_image_tuple:
            for i in range(len(img)):
                img[i].requires_grad = True
        else:
            img.requires_grad = True
        if record_interval != -1:
            img_list.append(deepcopy(tuple(to_np_array(*img, keep_list=True)) if args.is_image_tuple else to_np_array(img)))

    neg_mask_list = []
    if "mask" in args.ebm_target.split("+"):
        if neg_mask is None:
            img_shape = img[0].shape[2:] if args.is_image_tuple else img.shape[2:]
            w_dim = 1 if "mask" in model.w_type else in_channels
            neg_mask = tuple(torch.rand(batch_size, w_dim, *img_shape, device=device) for _ in range(model.mask_arity))
        for i in range(model.mask_arity):
            neg_mask[i].requires_grad = True
        if record_interval != -1:
            neg_mask_list.append(deepcopy(tuple(to_np_array(*neg_mask, keep_list=True))))

    c_repr_list = []
    if "repr" in args.ebm_target.split("+"):
        c_repr = torch.rand(batch_size, REPR_DIM, device=device) if c_repr is None else c_repr.to(device)
        c_repr.requires_grad = True
        if record_interval != -1:
            c_repr_list.append(deepcopy(to_np_array(c_repr)))

    z_list = []
    if "z" in args.ebm_target.split("+"):
        z_len = 1
        z = tuple(torch.rand(batch_size, model.z_dim, device=device) for _ in range(z_len)) if z is None else tuple(to_device_recur(z_ele, device) for z_ele in z)
        z_len = len(z)
        for i in range(z_len):
            if z[i] is not None:
                z[i].requires_grad = True
        if record_interval != -1:
            z_list.append(deepcopy(tuple(to_np_array(*z, keep_list=True))))
    assert z is None or isinstance(z, tuple) or isinstance(z, list)

    zgnn_list = []
    if "zgnn" in args.ebm_target.split("+"):
        n_nodes = len(z)
        n_edges = n_nodes * (n_nodes - 1)
        zgnn_dim = model.zgnn_dim
        zgnn = (torch.rand(batch_shape[0], n_nodes, zgnn_dim, device=device) if model.gnn.is_zgnn_node else None, torch.rand(batch_shape[0], n_edges, model.edge_attr_size, device=device)) if zgnn is None else tuple(to_device_recur(zgnn_ele, device) for zgnn_ele in zgnn)
        zgnn_len = len(zgnn)
        for i in range(zgnn_len):
            if zgnn[i] is not None:
                zgnn[i].requires_grad = True
        if record_interval != -1:
            zgnn_list.append(deepcopy(tuple(to_np_array(*zgnn, keep_list=True))))

    wtarget_list = []
    if "wtarget" in args.ebm_target.split("+"):
        w_dim = 1 if "mask" in model.w_type else in_channels
        if wtarget is None:
            wtarget = torch.rand(batch_size, w_dim, *img_shape, device=device)
        wtarget.requires_grad = True
        if record_interval != -1:
            wtarget_list.append(deepcopy(to_np_array(wtarget)))

    # Setting up noise and step_size scheduling:
    if args.lambd_start == -1:
        args.lambd_start = args.lambd
    lambd_list = args.lambd + 1/2 * (args.lambd_start - args.lambd) * (1 + torch.cos(torch.arange(args.sample_step)/args.sample_step * np.pi))
    if args.step_size_start == -1:
        args.step_size_start = args.step_size
    step_size_list = args.step_size + 1/2 * (args.step_size_start - args.step_size) * (1 + torch.cos(torch.arange(args.sample_step)/args.sample_step * np.pi))
    if args.SGLD_is_anneal:
        multiplier_list = np.linspace(0, 1, args.sample_step) ** args.SGLD_anneal_power
    else:
        multiplier_list = np.ones(args.sample_step)

    if args.SGLD_object_exceed_coef > 0:
        if isinstance(img, tuple):
            pos_all_mask = ((img[0][:,:1] == 1).float(), (img[1][:,:1] == 1).float())
        else:
            pos_all_mask = (img[:,:1] == 1).float()

    reg_dict = {}

    # SGLD:
    for k in range(args.sample_step):
        multiplier = multiplier_list[k]
        # Each step add noise using Langevin dynamics:
        if "image" in args.ebm_target.split("+"):
            if "noise_img" not in locals():
                noise_img = (torch.randn(batch_size, in_channels, *img_shape, device=device), torch.randn(batch_size, in_channels, *img_shape, device=device)) if args.is_image_tuple else torch.randn(batch_size, in_channels, *img_shape, device=device)
            if lambd_list[k] > 0:
                if args.is_image_tuple:
                    noise_img[0].normal_(0, lambd_list[k])
                    noise_img[1].normal_(0, lambd_list[k])
                    img[0].data.add_(noise_img[0].data)
                    img[1].data.add_(noise_img[1].data)
                else:
                    noise_img.normal_(0, lambd_list[k])
                    img.data.add_(noise_img.data)
        if "mask" in args.ebm_target.split("+"):
            if "noise" not in locals():
                w_dim = 1 if "mask" in model.w_type else in_channels
                noise = tuple(torch.randn(batch_size, w_dim, *(args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size)), device=device) for _ in range(model.mask_arity))
            if lambd_list[k] > 0:
                for i in range(model.mask_arity):
                    noise[i].normal_(0, lambd_list[k])
                    neg_mask[i].data.add_(noise[i].data)
        if "repr" in args.ebm_target.split("+"):
            if "noise_repr" not in locals():
                noise_repr = torch.randn(batch_size, REPR_DIM, device=device)
            if lambd_list[k] > 0:
                noise_repr.normal_(0, lambd_list[k])
                c_repr.data.add_(noise_repr.data)
        if "z" in args.ebm_target.split("+"):
            if "noise_z" not in locals():
                noise_z = tuple(torch.randn(batch_size, model.z_dim, device=device) if z[i] is not None else None for i in range(z_len))
            if lambd_list[k] > 0:
                for i in range(z_len):
                    if z[i] is not None:
                        noise_z[i].normal_(0, lambd_list[k])
                        z[i].data.add_(noise_z[i].data)
        if "zgnn" in args.ebm_target.split("+"):
            if "noise_zgnn" not in locals():
                noise_zgnn = tuple(torch.randn(zgnn[i].shape, device=device) if zgnn[i] is not None else None for i in range(zgnn_len))
            if lambd_list[k] > 0:
                for i in range(zgnn_len):
                    if zgnn[i] is not None:
                        noise_zgnn[i].normal_(0, lambd_list[k])
                        zgnn[i].data.add_(noise_zgnn[i].data)
        if "wtarget" in args.ebm_target.split("+"):
            if "noise_wtarget" not in locals():
                noise_wtarget = torch.randn(wtarget.shape, device=device)
            if lambd_list[k] > 0:
                noise_wtarget.normal_(0, lambd_list[k])
                wtarget.data.add_(noise_wtarget.data)

        # Compute the energy for each (pos_image, neg_mask, concept) instance:
        if k == args.sample_step - 1 and is_return_E:
            neg_out = model(img, neg_mask, c_repr=c_repr, z=z, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape, is_E_tensor=is_return_E)
            E_all = model.info["E_all"]
        else:
            neg_out = model(img, neg_mask, c_repr=c_repr, z=z, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape)
        neg_out_sum = neg_out.sum()
        record_data(reg_dict, deepcopy(to_np_array(neg_out)), "neg_out_core_list")
        if hasattr(model, "info"):
            for key in model.info:
                record_data(reg_dict, deepcopy(to_np_array(model.info[key])), "{}_list".format(key))

        # Penalize areas where the masks overlap with each other
        if args.SGLD_mutual_exclusive_coef > 0 and "mask" in args.ebm_target.split("+"):
            mutual_exclusive = get_neg_mask_overlap(neg_mask, mask_info=mask_info, is_penalize_lower=args.SGLD_is_penalize_lower, img=img) * args.SGLD_mutual_exclusive_coef * multiplier
            neg_out_sum += mutual_exclusive.sum()
            record_data(reg_dict, deepcopy(to_np_array(mutual_exclusive)), "mutual_exclusive_list")
        if args.SGLD_fine_mutual_exclusive_coef > 0 and "mask" in args.ebm_target.split("+"):
            fine_mutual_exclusive = get_fine_neg_mask_overlap(neg_mask, mask_info=mask_info) * args.SGLD_fine_mutual_exclusive_coef * multiplier
            neg_out_sum += fine_mutual_exclusive.sum()
            record_data(reg_dict, deepcopy(to_np_array(fine_mutual_exclusive)), "fine_mutual_exclusive_list")
        # Penalize areas where the negative mask hits areas not covered by the true mask
        if args.SGLD_object_exceed_coef > 0 and "mask" in args.ebm_target.split("+"):
            # SGLD_object_exceed_coef is annealed quadratically as the sample steps continue
            object_exceed = get_neg_mask_exceed(neg_mask, pos_all_mask) * args.SGLD_object_exceed_coef * multiplier
            neg_out_sum += object_exceed.sum()
            record_data(reg_dict, deepcopy(to_np_array(object_exceed)), "object_exceed_list")
        # Encourage each EBM will either fully explain or fully ignore each pixel:
        if args.SGLD_pixel_entropy_coef > 0 and "mask" in args.ebm_target.split("+"):
            pixel_entropy = get_pixel_entropy(neg_mask) * args.SGLD_pixel_entropy_coef * multiplier
            neg_out_sum += pixel_entropy.sum()
            record_data(reg_dict, deepcopy(to_np_array(pixel_entropy)), "pixel_entropy_list")
        # Encourage each EBM mask will either fully explain an object or not explain any
        if args.SGLD_mask_entropy_coef > 0 and "mask" in args.ebm_target.split("+"):
            mask_entropy = get_mask_entropy(neg_mask) * args.SGLD_mask_entropy_coef * multiplier
            neg_out_sum += mask_entropy.sum()
            record_data(reg_dict, deepcopy(to_np_array(mask_entropy)), "mask_entropy_list")
        # Encourage the EBMs to specialize, where the ebms whose masks that are nearest to 0 or 1 will be push hardes towards 0 or 1:
        if args.SGLD_pixel_gm_coef > 0 and "mask" in args.ebm_target.split("+"):
            pixel_gm = get_pixel_gm(neg_mask) * args.SGLD_pixel_gm_coef * multiplier
            neg_out_sum += pixel_gm.sum()
            record_data(reg_dict, deepcopy(to_np_array(pixel_gm)), "pixel_gm_list")
        if "mask" in args.ebm_target.split("+") and (
            args.SGLD_iou_batch_consistency_coef > 0 or
            args.SGLD_iou_concept_repel_coef > 0 or
            args.SGLD_iou_relation_repel_coef > 0 or
            args.SGLD_iou_relation_overlap_coef > 0 or
            args.SGLD_iou_attract_coef > 0):
            graph_energy = get_graph_energy(
                neg_mask,
                mask_info=mask_info,
                iou_batch_consistency_coef=args.SGLD_iou_batch_consistency_coef,
                iou_concept_repel_coef=args.SGLD_iou_concept_repel_coef,
                iou_relation_repel_coef=args.SGLD_iou_relation_repel_coef,
                iou_relation_overlap_coef=args.SGLD_iou_relation_overlap_coef,
                iou_attract_coef=args.SGLD_iou_attract_coef,
                batch_shape=batch_shape,
            )[0] * multiplier
            neg_out_sum += graph_energy.sum()
            record_data(reg_dict, deepcopy(to_np_array(graph_energy)), "graph_energy_list")

        neg_out_sum.backward()

        # Perform gradient descent on the neg_mask(s) and/or c_repr:
        if "image" in args.ebm_target.split("+"):
            if args.is_image_tuple:
                for i in range(len(img)):
                    img[i].grad.data.clamp_(-0.01, 0.01)
                    img[i].data.add_(img[i].grad.data, alpha=-args.step_size_img)
                    img[i].grad.detach_()
                    img[i].grad.zero_()
                    img[i].data.clamp_(img_value_min, img_value_max)
                if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                    img_list.append(deepcopy(tuple(to_np_array(*img, keep_list=True))))
            else:
                img.grad.data.clamp_(-0.01, 0.01)
                img.data.add_(img.grad.data, alpha=-args.step_size_img)
                img.grad.detach_()
                img.grad.zero_()
                img.data.clamp_(img_value_min, img_value_max)
                if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                    img_list.append(deepcopy(to_np_array(img)))
        if "mask" in args.ebm_target.split("+"):
            for i in range(model.mask_arity):
                neg_mask[i].grad.data.clamp_(-0.01, 0.01)
                neg_mask[i].data.add_(neg_mask[i].grad.data, alpha=-step_size_list[k])
                neg_mask[i].grad.detach_()
                neg_mask[i].grad.zero_()
                neg_mask[i].data.clamp_(0, 1)
            if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                neg_mask_list.append(deepcopy(tuple(to_np_array(*neg_mask, keep_list=True))))
        if "repr" in args.ebm_target.split("+"):
            c_repr.grad.data.clamp_(-0.01, 0.01)
            c_repr.data.add_(c_repr.grad.data, alpha=-args.step_size_repr)
            c_repr.grad.detach_()
            c_repr.grad.zero_()
            if "softmax" not in args.c_repr_mode:
                c_repr.data.clamp_(0, 1)
            if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                c_repr_list.append(deepcopy(to_np_array(c_repr)))
        if "z" in args.ebm_target.split("+"):
            for i in range(len(z)):
                if z[i] is not None:
                    z[i].grad.data.clamp_(-0.01, 0.01)
                    z[i].data.add_(z[i].grad.data, alpha=-args.step_size_z)
                    z[i].grad.detach_()
                    z[i].grad.zero_()
                    z[i].data.clamp_(0, 1)
            if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                z_list.append(deepcopy(tuple(to_np_array(*z, keep_list=True))))
        if "zgnn" in args.ebm_target.split("+"):
            for i in range(zgnn_len):
                if zgnn[i] is not None:
                    zgnn[i].grad.data.clamp_(-0.01, 0.01)
                    zgnn[i].data.add_(zgnn[i].grad.data, alpha=-args.step_size_zgnn)
                    zgnn[i].grad.detach_()
                    zgnn[i].grad.zero_()
                    zgnn[i].data.clamp_(0, 1)
            if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                zgnn_list.append(deepcopy(tuple(to_np_array(*zgnn, keep_list=True))))
        if "wtarget" in args.ebm_target.split("+"):
            wtarget.grad.data.clamp_(-0.01, 0.01)
            wtarget.data.add_(wtarget.grad.data, alpha=-args.step_size_wtarget)
            wtarget.grad.detach_()
            wtarget.grad.zero_()
            wtarget.data.clamp_(0, 1)
            if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                wtarget_list.append(deepcopy(tuple(to_np_array(*wtarget, keep_list=True))))

        neg_out_list.append(deepcopy(to_np_array(neg_out)))

    """
    Record the energies of the last step:
    """
    neg_out = model(img, neg_mask, c_repr=c_repr, z=z, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape)
    neg_out_list.append(deepcopy(to_np_array(neg_out)))
    if hasattr(model, "info"):
        for key in model.info:
            record_data(reg_dict, deepcopy(to_np_array(model.info[key])), "{}_list".format(key))
    # Penalize areas where the masks overlap with each other
    if args.SGLD_mutual_exclusive_coef > 0 and "mask" in args.ebm_target.split("+"):
        mutual_exclusive = get_neg_mask_overlap(neg_mask, mask_info=mask_info, is_penalize_lower=args.SGLD_is_penalize_lower, img=img) * args.SGLD_mutual_exclusive_coef * multiplier
        record_data(reg_dict, deepcopy(to_np_array(mutual_exclusive)), "mutual_exclusive_list")
    if args.SGLD_fine_mutual_exclusive_coef > 0 and "mask" in args.ebm_target.split("+"):
        fine_mutual_exclusive = get_fine_neg_mask_overlap(neg_mask, mask_info=mask_info) * args.SGLD_fine_mutual_exclusive_coef * multiplier
        record_data(reg_dict, deepcopy(to_np_array(fine_mutual_exclusive)), "fine_mutual_exclusive_list")
    # Penalize areas where the negative mask hits areas not covered by the true mask
    if args.SGLD_object_exceed_coef > 0 and "mask" in args.ebm_target.split("+"):
        # SGLD_object_exceed_coef is annealed quadratically as the sample steps continue
        object_exceed = get_neg_mask_exceed(neg_mask, pos_all_mask) * args.SGLD_object_exceed_coef * multiplier
        record_data(reg_dict, deepcopy(to_np_array(object_exceed)), "object_exceed_list")
    # Encourage each EBM will either fully explain or fully ignore each pixel:
    if args.SGLD_pixel_entropy_coef > 0 and "mask" in args.ebm_target.split("+"):
        pixel_entropy = get_pixel_entropy(neg_mask) * args.SGLD_pixel_entropy_coef * multiplier
        record_data(reg_dict, deepcopy(to_np_array(pixel_entropy)), "pixel_entropy_list")
    # Encourage each EBM mask will either fully explain an object or not explain any
    if args.SGLD_mask_entropy_coef > 0 and "mask" in args.ebm_target.split("+"):
        mask_entropy = get_mask_entropy(neg_mask) * args.SGLD_mask_entropy_coef * multiplier
        record_data(reg_dict, deepcopy(to_np_array(mask_entropy)), "mask_entropy_list")
    # Encourage the EBMs to specialize, where the ebms whose masks that are nearest to 0 or 1 will be push hardes towards 0 or 1:
    if args.SGLD_pixel_gm_coef > 0 and "mask" in args.ebm_target.split("+"):
        pixel_gm = get_pixel_gm(neg_mask) * args.SGLD_pixel_gm_coef * multiplier
        record_data(reg_dict, deepcopy(to_np_array(pixel_gm)), "pixel_gm_list")
    if "mask" in args.ebm_target.split("+") and (
        args.SGLD_iou_batch_consistency_coef > 0 or
        args.SGLD_iou_concept_repel_coef > 0 or
        args.SGLD_iou_relation_repel_coef > 0 or
        args.SGLD_iou_relation_overlap_coef > 0 or
        args.SGLD_iou_attract_coef > 0):
        graph_energy = get_graph_energy(
            neg_mask,
            mask_info=mask_info,
            iou_batch_consistency_coef=args.SGLD_iou_batch_consistency_coef,
            iou_concept_repel_coef=args.SGLD_iou_concept_repel_coef,
            iou_relation_repel_coef=args.SGLD_iou_relation_repel_coef,
            iou_relation_overlap_coef=args.SGLD_iou_relation_overlap_coef,
            iou_attract_coef=args.SGLD_iou_attract_coef,
            batch_shape=batch_shape,
        )[0] * multiplier
        record_data(reg_dict, deepcopy(to_np_array(graph_energy)), "graph_energy_list")

    neg_out_list = np.concatenate(neg_out_list, -1).T   # after: [sample_step, B]

    if "image" in args.ebm_target.split("+"):
        if args.is_image_tuple:
            img = tuple(img[i].detach() for i in range(len(img)))
            if record_interval != -1:
                img_list = tuple(Zip(*img_list, function=np.stack))
        else:
            img = img.detach()
            if record_interval != -1:
                img_list = np.stack(img_list)
    if "mask" in args.ebm_target.split("+"):
        neg_mask = tuple(neg_mask[i].detach() for i in range(model.mask_arity))
        if record_interval != -1:
            neg_mask_list = tuple(Zip(*neg_mask_list, function=np.stack))
    if "repr" in args.ebm_target.split("+"):
        c_repr = c_repr.detach()
        if record_interval != -1:
            c_repr_list = np.stack(c_repr_list)
    if "z" in args.ebm_target.split("+"):
        z = tuple(z[i].detach() if z[i] is not None else None for i in range(z_len))
        if record_interval != -1:
            z_list = tuple(Zip(*z_list, function=np.stack))
    if "zgnn" in args.ebm_target.split("+"):
        zgnn = tuple(zgnn[i].detach() if zgnn[i] is not None else None for i in range(zgnn_len))
        if record_interval != -1:
            zgnn_list = tuple(Zip(*zgnn_list))
    if "wtarget" in args.ebm_target.split("+"):
        wtarget = wtarget.detach()
        if record_interval != -1:
            wtarget_list = np.stack(wtarget_list)

    info = {
        "neg_out_list": neg_out_list,
        "img_list": img_list,
        "neg_mask_list": neg_mask_list,
        "c_repr_list": c_repr_list,
        "z_list": z_list,
        "zgnn_list": zgnn_list,
        "wtarget_list": wtarget_list,
    }
    reg_dict = transform_dict(reg_dict, "array")
    info.update(reg_dict)
    if is_return_E:
        info["E_all"] = E_all
    return (img, neg_mask, c_repr, z, zgnn, wtarget), info


# ### 1.3.3 SGLD with KL:

# In[ ]:


def neg_mask_sgd_with_kl(
    model,
    img=None,
    neg_mask=None,
    c_repr=None,
    z=None,
    zgnn=None,
    wtarget=None,
    args=None,
    mask_info=None,
    is_return_E=False,
    batch_shape=None,
    record_interval=-1,
):
    """Perform SGLD w.r.t. a subset of {img, neg_mask, c_repr, z} given the others, and in addition compute the the same thing for kl."""
    if args.step_size_img == -1:
        args.step_size_img = args.step_size
    if args.step_size_z == -1:
        args.step_size_z = args.step_size
    if args.step_size_zgnn == -1: 
        args.step_size_zgnn = args.step_size
    if args.step_size_wtarget == -1:
        args.step_size_wtarget = args.step_size
    neg_out_list = []
    # Initialize neg_mask and c_repr:
    if img is not None:
        batch_size = img[0].shape[0] if args.is_image_tuple else img.shape[0]
        in_channels = img[0].shape[1] if args.is_image_tuple else img.shape[1]
        device = img[0].device if args.is_image_tuple else img.device
        img_shape = img[0].shape[2:] if args.is_image_tuple else img.shape[2:]
    else:
        batch_size = neg_mask[0].shape[0]
        in_channels = args.in_channels
        img_shape = args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size)
        device = neg_mask[0].device

    img_list = []
    if "image" in args.ebm_target.split("+"):
        img_value_min, img_value_max = args.image_value_range.split(",")
        img_value_min, img_value_max = eval(img_value_min), eval(img_value_max)
        img_value_span = img_value_max - img_value_min
        assert img_value_span >= 1
        if img is None:
            img = (
                torch.rand(batch_size, in_channels, *img_shape, device=device) * img_value_span + img_value_min, torch.rand(batch_size, in_channels, *img_shape, device=device) * img_value_span + img_value_min
            ) if args.is_image_tuple else torch.rand(batch_size, in_channels, *img_shape, device=device) * img_value_span + img_value_min
        else:
            if isinstance(img, tuple):
                img = tuple(torch.rand(batch_size, in_channels, *img_shape, device=device) * img_value_span + img_value_min if img_ele is None else img_ele for img_ele in img)
        if record_interval != -1:
            img_list.append(deepcopy(tuple(to_np_array(*img, keep_list=True)) if args.is_image_tuple else to_np_array(img)))

    neg_mask_list = []
    if "mask" in args.ebm_target.split("+"):
        w_dim = 1 if "mask" in model.w_type else in_channels
        if neg_mask is None:
            neg_mask = tuple(torch.rand(batch_size, w_dim, *img_shape, device=device) for _ in range(model.mask_arity))
        if record_interval != -1:
            neg_mask_list.append(deepcopy(tuple(to_np_array(*neg_mask, keep_list=True))))

    c_repr_list = []
    if "repr" in args.ebm_target.split("+"):
        c_repr = torch.rand(batch_size, REPR_DIM, device=device) if c_repr is None else c_repr
        if record_interval != -1:
            c_repr_list.append(deepcopy(to_np_array(c_repr)))

    z_list = []
    if "z" in args.ebm_target.split("+"):
        z_len = 1
        z = tuple(torch.rand(batch_size, model.z_dim, device=device) for _ in range(z_len)) if z is None else tuple(to_device_recur(z_ele, device) for z_ele in z)
        z_len = len(z)
        if record_interval != -1:
            z_list.append(deepcopy(tuple(to_np_array(*z, keep_list=True))))
    assert z is None or isinstance(z, tuple) or isinstance(z, list)

    zgnn_list = []
    if "zgnn" in args.ebm_target.split("+"):
        n_nodes = len(z) if z is not None else len(neg_mask)
        n_edges = n_nodes * (n_nodes - 1)
        zgnn_dim = model.zgnn_dim
        zgnn = (torch.rand(batch_shape[0], n_nodes, zgnn_dim, device=device) if model.gnn.is_zgnn_node else None, torch.rand(batch_shape[0], n_edges, model.edge_attr_size, device=device)) if zgnn is None else tuple(to_device_recur(zgnn_ele, device) for zgnn_ele in zgnn)
        zgnn_len = len(zgnn)
        if record_interval != -1:
            zgnn_list.append(deepcopy(tuple(to_np_array(*zgnn, keep_list=True))))

    wtarget_list = []
    if "wtarget" in args.ebm_target.split("+"):
        w_dim = 1 if "mask" in model.w_type else in_channels
        if wtarget is None:
            wtarget = torch.rand(batch_size, w_dim, *img_shape, device=device)
        if record_interval != -1:
            wtarget_list.append(deepcopy(to_np_array(wtarget)))

    # Setting up noise and step_size scheduling:
    if args.lambd_start == -1:
        args.lambd_start = args.lambd
    lambd_list = args.lambd + 1/2 * (args.lambd_start - args.lambd) * (1 + torch.cos(torch.arange(args.sample_step)/args.sample_step * np.pi))
    if args.step_size_start == -1:
        args.step_size_start = args.step_size
    step_size_list = args.step_size + 1/2 * (args.step_size_start - args.step_size) * (1 + torch.cos(torch.arange(args.sample_step)/args.sample_step * np.pi))
    if args.SGLD_is_anneal:
        multiplier_list = np.linspace(0, 1, args.sample_step) ** args.SGLD_anneal_power
    else:
        multiplier_list = np.ones(args.sample_step)

    if args.SGLD_object_exceed_coef > 0:
        if isinstance(img, tuple):
            pos_all_mask = ((img[0][:,:1] == 1).float(), (img[1][:,:1] == 1).float())
        else:
            pos_all_mask = (img[:,:1] == 1).float()

    reg_dict = {}

    # SGLD:
    for k in range(args.sample_step):
        multiplier = multiplier_list[k]
        # Each step add noise using Langevin dynamics:
        if "image" in args.ebm_target.split("+"):
            if "noise_img" not in locals():
                noise_img = (torch.randn(batch_size, in_channels, *img_shape, device=device), torch.randn(batch_size, in_channels, *img_shape, device=device)) if args.is_image_tuple else torch.randn(batch_size, in_channels, *img_shape, device=device)
            if lambd_list[k] > 0:
                if args.is_image_tuple:
                    for i in range(len(img)):
                        noise_img[i].normal_(0, lambd_list[k])
                    img = tuple(img[i] + noise_img[i] for i in range(len(img)))
                else:
                    noise_img.normal_(0, lambd_list[k])
                    img = img + noise_img
            if args.is_image_tuple:
                for i in range(len(img)):
                    img[i].requires_grad_(True)
            else:
                img.requires_grad_(True)
        if "mask" in args.ebm_target.split("+"):
            if "noise" not in locals():
                w_dim = 1 if "mask" in model.w_type else in_channels
                noise = tuple(torch.randn(batch_size, w_dim, *(args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size)), device=device) for _ in range(model.mask_arity))
            if lambd_list[k] > 0:
                for i in range(model.mask_arity):
                    noise[i].normal_(0, lambd_list[k])
                neg_mask = tuple(neg_mask[i] + noise[i] for i in range(model.mask_arity))
            for i in range(model.mask_arity):
                neg_mask[i].requires_grad_(True)
        if "repr" in args.ebm_target.split("+"):
            if "noise_repr" not in locals():
                noise_repr = torch.randn(batch_size, REPR_DIM, device=device)
            if lambd_list[k] > 0:
                noise_repr.normal_(0, lambd_list[k])
                c_repr = c_repr + noise_repr
            c_repr.requires_grad_(True)
        if "z" in args.ebm_target.split("+"):
            if "noise_z" not in locals():
                noise_z = tuple(torch.randn(batch_size, model.z_dim, device=device) if z[i] is not None else None for i in range(z_len))
            if lambd_list[k] > 0:
                for i in range(z_len):
                    if z[i] is not None:
                        noise_z[i].normal_(0, lambd_list[k])
                z = tuple(z[i] + noise_z[i] if z[i] is not None else None for i in range(z_len))
            for i in range(z_len):
                if z[i] is not None:
                    z[i].requires_grad_(True)
        if "zgnn" in args.ebm_target.split("+"):
            if "noise_zgnn" not in locals():
                noise_zgnn = tuple(torch.randn(zgnn[i].shape, device=device) if zgnn[i] is not None else None for i in range(zgnn_len))
            if lambd_list[k] > 0:
                for i in range(zgnn_len):
                    if zgnn[i] is not None:
                        noise_zgnn[i].normal_(0, lambd_list[k])
                zgnn = tuple(zgnn[i] + noise_zgnn[i] if zgnn[i] is not None else None for i in range(zgnn_len))
            for i in range(zgnn_len):
                if zgnn[i] is not None:
                    zgnn[i].requires_grad_(True)
        if "wtarget" in args.ebm_target.split("+"):
            if "noise_wtarget" not in locals():
                w_dim = 1 if "mask" in model.w_type else in_channels
                noise_wtarget = torch.randn(batch_size, w_dim, *(args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size)), device=device)
            if lambd_list[k] > 0:
                noise_wtarget.normal_(0, lambd_list[k])
                wtarget = wtarget + noise_wtarget
            wtarget.requires_grad_(True)

        # Compute neg_out and the gradient:
        neg_out = model(img, neg_mask, c_repr=c_repr, z=z, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape)
        neg_out_sum = neg_out.sum()
        record_data(reg_dict, deepcopy(to_np_array(neg_out)), "neg_out_core_list")

        # Penalize areas where the masks overlap with each other
        if args.SGLD_mutual_exclusive_coef > 0 and "mask" in args.ebm_target.split("+"):
            mutual_exclusive = get_neg_mask_overlap(neg_mask, mask_info=mask_info, is_penalize_lower=args.SGLD_is_penalize_lower, img=img) * args.SGLD_mutual_exclusive_coef * multiplier
            neg_out_sum += mutual_exclusive.sum()
            record_data(reg_dict, deepcopy(to_np_array(mutual_exclusive)), "mutual_exclusive_list")
        if args.SGLD_fine_mutual_exclusive_coef > 0 and "mask" in args.ebm_target.split("+"):
            fine_mutual_exclusive = get_fine_neg_mask_overlap(neg_mask, mask_info=mask_info) * args.SGLD_fine_mutual_exclusive_coef * multiplier
            neg_out_sum += fine_mutual_exclusive.sum()
            record_data(reg_dict, deepcopy(to_np_array(fine_mutual_exclusive)), "fine_mutual_exclusive_list")
        # Penalize areas where the negative mask hits areas not covered by the true mask
        if args.SGLD_object_exceed_coef > 0 and "mask" in args.ebm_target.split("+"):
            # SGLD_object_exceed_coef is annealed quadratically as the sample steps continue
            object_exceed = get_neg_mask_exceed(neg_mask, pos_all_mask) * args.SGLD_object_exceed_coef * multiplier
            neg_out_sum += object_exceed.sum()
            record_data(reg_dict, deepcopy(to_np_array(object_exceed)), "object_exceed_list")
        # Encourage each EBM will either fully explain or fully ignore each pixel:
        if args.SGLD_pixel_entropy_coef > 0 and "mask" in args.ebm_target.split("+"):
            pixel_entropy = get_pixel_entropy(neg_mask) * args.SGLD_pixel_entropy_coef * multiplier
            neg_out_sum += pixel_entropy.sum()
            record_data(reg_dict, deepcopy(to_np_array(pixel_entropy)), "pixel_entropy_list")
        # Encourage each EBM mask will either fully explain an object or not explain any:
        if args.SGLD_mask_entropy_coef > 0 and "mask" in args.ebm_target.split("+"):
            mask_entropy = get_mask_entropy(neg_mask) * args.SGLD_mask_entropy_coef * multiplier
            neg_out_sum += mask_entropy.sum()
            record_data(reg_dict, deepcopy(to_np_array(mask_entropy)), "mask_entropy_list")
        # Encourage the ebms to specialize, where the ebms whose masks that are nearest to 0 or 1 will be push harder towards 0 or 1:
        if args.SGLD_pixel_gm_coef > 0 and "mask" in args.ebm_target.split("+"):
            pixel_gm = get_pixel_gm(neg_mask) * args.SGLD_pixel_gm_coef * multiplier
            neg_out_sum += pixel_gm.sum()
            record_data(reg_dict, deepcopy(to_np_array(pixel_gm)), "pixel_gm_list")
        # Compute the energies that encourages selector discovery:
        if "mask" in args.ebm_target.split("+") and (
            args.SGLD_iou_batch_consistency_coef > 0 or
            args.SGLD_iou_concept_repel_coef > 0 or
            args.SGLD_iou_relation_repel_coef > 0 or
            args.SGLD_iou_relation_overlap_coef > 0 or
            args.SGLD_iou_attract_coef > 0):
            graph_energy = get_graph_energy(
                neg_mask,
                mask_info=mask_info,
                iou_batch_consistency_coef=args.SGLD_iou_batch_consistency_coef,
                iou_concept_repel_coef=args.SGLD_iou_concept_repel_coef,
                iou_relation_repel_coef=args.SGLD_iou_relation_repel_coef,
                iou_relation_overlap_coef=args.SGLD_iou_relation_overlap_coef,
                iou_attract_coef=args.SGLD_iou_attract_coef,
                batch_shape=batch_shape,
            )[0] * multiplier
            neg_out_sum += graph_energy.sum()
            record_data(reg_dict, deepcopy(to_np_array(graph_energy)), "graph_energy_list")

        if "image" in args.ebm_target.split("+"):
            if args.is_image_tuple:
                img_grad = tuple(torch.autograd.grad([neg_out_sum],
                                  [img[i]],
                                  create_graph=True if args.kl_all_step else False,
                                  retain_graph=True if "mask" in args.ebm_target.split("+") or "repr" in args.ebm_target.split("+") or "z" in args.ebm_target.split("+") or "zgnn" in args.ebm_target.split("+") or "wtarget" in args.ebm_target.split("+") or i < len(img) - 1 else None)[0] for i in range(len(img)))
            else:
                img_grad = torch.autograd.grad([neg_out_sum],
                                               [img],
                                               create_graph=True if args.kl_all_step else False,
                                               retain_graph=True if "mask" in args.ebm_target.split("+") or "repr" in args.ebm_target.split("+") or "z" in args.ebm_target.split("+") or "zgnn" in args.ebm_target.split("+") or "wtarget" in args.ebm_target.split("+") else None)[0]
        if "mask" in args.ebm_target.split("+"):
            neg_mask_grad = tuple(torch.autograd.grad([neg_out_sum],
                                  [neg_mask[i]],
                                  create_graph=True if args.kl_all_step else False,
                                  retain_graph=True if "repr" in args.ebm_target.split("+") or "z" in args.ebm_target.split("+") or "zgnn" in args.ebm_target.split("+") or "wtarget" in args.ebm_target.split("+") or i < model.mask_arity - 1 else None)[0] for i in range(model.mask_arity))
        if "repr" in args.ebm_target.split("+"):
            c_repr_grad = torch.autograd.grad([neg_out_sum], [c_repr], create_graph=True if args.kl_all_step else False,
                                              retain_graph=True if "z" in args.ebm_target.split("+") or "zgnn" in args.ebm_target.split("+") or "wtarget" in args.ebm_target.split("+") else None
                                             )[0]
        if "z" in args.ebm_target.split("+"):
            z_grad = tuple(torch.autograd.grad([neg_out_sum],
                                               [z[i]],
                                               create_graph=True if args.kl_all_step else False,
                                               retain_graph=True if "zgnn" in args.ebm_target.split("+") or "wtarget" in args.ebm_target.split("+") or i < z_len - 1 else None)[0] if z[i] is not None else None for i in range(z_len))
        if "zgnn" in args.ebm_target.split("+"):
            zgnn_grad = tuple(torch.autograd.grad([neg_out_sum],
                                                  [zgnn[i]],
                                                  create_graph=True if args.kl_all_step else False,
                                                  retain_graph=True if "wtarget" in args.ebm_target.split("+") or i < zgnn_len - 1 else None)[0] if zgnn[i] is not None else None for i in range(zgnn_len))
        if "wtarget" in args.ebm_target.split("+"):
            wtarget_grad = torch.autograd.grad([neg_out_sum], [wtarget], create_graph=True if args.kl_all_step else False)[0]

        if k == args.sample_step - 1:
            img_ori = img
            neg_mask_ori = neg_mask
            c_repr_ori = c_repr
            z_ori = z
            zgnn_ori = zgnn
            wtarget_ori = wtarget
            # Update subset of {img, neg_mask, c_repr, z} using the gradient:
            if "image" in args.ebm_target.split("+"):
                if args.is_image_tuple:
                    img = tuple(img[i] - img_grad[i] * args.step_size_img for i in range(len(img)))

                    # Update the img_kl:
                    img_kl = img_ori
                    neg_out_kl_sum = model(img_kl, neg_mask, c_repr=c_repr, z=z, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape).sum()
                    img_kl_grad = tuple(torch.autograd.grad([neg_out_kl_sum], [img_kl[i]], create_graph=True)[0] for i in range(len(img)))
                    img_kl = tuple(img_kl[i] - img_kl_grad[i] * args.step_size_img for i in range(len(img)))
                    img_kl = tuple(torch.clamp(img_kl[i], img_value_min, img_value_max) for i in range(len(img)))

                    # Detach and clamp neg_mask:
                    img = tuple(img[i].detach() for i in range(len(img)))
                    img = tuple(torch.clamp(img[i], img_value_min, img_value_max) for i in range(len(img)))
                    if record_interval != -1:
                        img_list.append(deepcopy(tuple(to_np_array(*img, keep_list=True))))
                else:
                    img = img - img_grad * args.step_size_img

                    # Update the img_kl:
                    img_kl = img_ori
                    neg_out_kl_sum = model(img_kl, neg_mask, c_repr=c_repr, z=z, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape).sum()
                    img_kl_grad = torch.autograd.grad([neg_out_kl_sum], [img_kl], create_graph=True)[0]
                    img_kl = img_kl - img_kl_grad * args.step_size_img
                    img_kl = torch.clamp(img_kl, img_value_min, img_value_max)

                    # Detach and clamp z:
                    img = img.detach()
                    img = torch.clamp(img, img_value_min, img_value_max)
                    if record_interval != -1:
                        img_list.append(deepcopy(to_np_array(img)))

            if "mask" in args.ebm_target.split("+"):
                neg_mask = tuple(neg_mask[i] - neg_mask_grad[i] * step_size_list[k] for i in range(model.mask_arity))

                # Update the neg_mask_kl:
                neg_mask_kl = neg_mask_ori
                if is_return_E:
                    neg_out_kl_sum = model(img_ori, neg_mask_kl, c_repr=c_repr, z=z, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape, is_E_tensor=is_return_E).sum()
                    E_all = model.info["E_all"]
                else:
                    neg_out_kl_sum = model(img_ori, neg_mask_kl, c_repr=c_repr, z=z, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape).sum()

                if args.SGLD_mutual_exclusive_coef > 0:
                    neg_out_kl_sum += get_neg_mask_overlap(neg_mask_kl, mask_info=mask_info, is_penalize_lower=args.SGLD_is_penalize_lower, img=img).sum() * args.SGLD_mutual_exclusive_coef * multiplier
                if args.SGLD_fine_mutual_exclusive_coef > 0 and "mask" in args.ebm_target.split("+"):
                    neg_out_kl_sum += get_fine_neg_mask_overlap(neg_mask_kl, mask_info=mask_info).sum() * args.SGLD_fine_mutual_exclusive_coef * multiplier
                # Penalize areas where the negative mask hits areas not covered by the true mask
                if args.SGLD_object_exceed_coef > 0:
                    # SGLD_object_exceed_coef is annealed quadratically as the sample steps continue
                    neg_out_kl_sum += get_neg_mask_exceed(neg_mask_kl, pos_all_mask).sum() * args.SGLD_object_exceed_coef * multiplier
                # Encourage each EBM will either fully explain or fully ignore each pixel:
                if args.SGLD_pixel_entropy_coef > 0:
                    neg_out_kl_sum += get_pixel_entropy(neg_mask_kl).sum() * args.SGLD_pixel_entropy_coef * multiplier
                # Encourage each EBM mask will either fully explain an object or not explain any:
                if args.SGLD_mask_entropy_coef > 0:
                    neg_out_kl_sum += get_mask_entropy(neg_mask_kl).sum() * args.SGLD_mask_entropy_coef * multiplier
                # Encourage the ebms to specialize, where the ebms whose masks that are nearest to 0 or 1 will be push harder towards 0 or 1:
                if args.SGLD_pixel_gm_coef > 0:
                    neg_out_kl_sum += get_pixel_gm(neg_mask_kl).sum() * args.SGLD_pixel_gm_coef * multiplier
                # Compute the energies that encourages selector discovery:
                if "mask" in args.ebm_target.split("+") and (
                    args.SGLD_iou_batch_consistency_coef > 0 or
                    args.SGLD_iou_concept_repel_coef > 0 or
                    args.SGLD_iou_relation_repel_coef > 0 or
                    args.SGLD_iou_relation_overlap_coef > 0 or
                    args.SGLD_iou_attract_coef > 0):
                    neg_out_kl_sum += get_graph_energy(
                        neg_mask_kl,
                        mask_info=mask_info,
                        iou_batch_consistency_coef=args.SGLD_iou_batch_consistency_coef,
                        iou_concept_repel_coef=args.SGLD_iou_concept_repel_coef,
                        iou_relation_repel_coef=args.SGLD_iou_relation_repel_coef,
                        iou_relation_overlap_coef=args.SGLD_iou_relation_overlap_coef,
                        iou_attract_coef=args.SGLD_iou_attract_coef,
                        batch_shape=batch_shape,
                    )[0].sum() * multiplier

                neg_mask_kl_grad = tuple(torch.autograd.grad([neg_out_kl_sum], [neg_mask_kl[i]], create_graph=True)[0] for i in range(model.mask_arity))
                neg_mask_kl = tuple(neg_mask_kl[i] - neg_mask_kl_grad[i] * step_size_list[k] for i in range(model.mask_arity))
                neg_mask_kl = tuple(torch.clamp(neg_mask_kl[i], 0, 1) for i in range(model.mask_arity))

                # Detach and clamp neg_mask:
                neg_mask = tuple(neg_mask[i].detach() for i in range(model.mask_arity))
                neg_mask = tuple(torch.clamp(neg_mask[i], 0, 1) for i in range(model.mask_arity))
                if record_interval != -1:
                    neg_mask_list.append(deepcopy(tuple(to_np_array(*neg_mask, keep_list=True))))
            if "repr" in args.ebm_target.split("+"):
                c_repr = c_repr - c_repr_grad * args.step_size_repr

                # Update the c_repr_kl:
                c_repr_kl = c_repr_ori
                neg_out_kl_sum = model(img_ori, neg_mask_ori, c_repr=c_repr_kl, z=z, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape).sum()
                # No need for SGLD_mutual_exclusive_coef since the grad is w.r.t. c_repr_kl.
                c_repr_kl_grad = torch.autograd.grad([neg_out_kl_sum], [c_repr_kl], create_graph=True)[0]
                c_repr_kl = c_repr_kl - c_repr_kl_grad * args.step_size_repr
                if "softmax" not in args.c_repr_mode:
                    c_repr_kl = torch.clamp(c_repr_kl, 0, 1)

                # Detach and clamp c_repr:
                c_repr = c_repr.detach()
                if "softmax" not in args.c_repr_mode:
                    c_repr = torch.clamp(c_repr, 0, 1)
                if record_interval != -1:
                    c_repr_list.append(deepcopy(to_np_array(c_repr)))
            if "z" in args.ebm_target.split("+"):
                z = tuple(z[i] - z_grad[i] * args.step_size_z if z[i] is not None else None for i in range(z_len))

                # Update the z_kl:
                z_kl = z_ori
                neg_out_kl_sum = model(img_ori, neg_mask_ori, c_repr=c_repr_ori, z=z_kl, zgnn=zgnn, wtarget=wtarget, batch_shape=batch_shape).sum()
                z_kl_grad = tuple(torch.autograd.grad([neg_out_kl_sum], [z_kl[i]], create_graph=True)[0] if z[i] is not None else None for i in range(z_len))
                z_kl = tuple(z_kl[i] - z_kl_grad[i] * args.step_size_z if z[i] is not None else None for i in range(z_len))
                z_kl = tuple(torch.clamp(z_kl[i], 0, 1) if z[i] is not None else None for i in range(z_len))

                # Detach and clamp z:
                z = tuple(z[i].detach() if z[i] is not None else None for i in range(z_len))
                z = tuple(torch.clamp(z[i], 0, 1) if z[i] is not None else None for i in range(z_len))
                if record_interval != -1:
                    z_list.append(deepcopy(tuple(to_np_array(*z, keep_list=True))))
            if "zgnn" in args.ebm_target.split("+"):
                zgnn = tuple(zgnn[i] - zgnn_grad[i] * args.step_size_zgnn if zgnn[i] is not None else None for i in range(zgnn_len))

                # Update the zgnn_kl:
                zgnn_kl = zgnn_ori
                neg_out_kl_sum = model(img_ori, neg_mask_ori, c_repr=c_repr_ori, z=z_ori, zgnn=zgnn_kl, wtarget=wtarget, batch_shape=batch_shape).sum()
                zgnn_kl_grad = tuple(torch.autograd.grad([neg_out_kl_sum], [zgnn_kl[i]], create_graph=True)[0] if zgnn[i] is not None else None for i in range(zgnn_len))
                zgnn_kl = tuple(zgnn_kl[i] - zgnn_kl_grad[i] * args.step_size_zgnn if zgnn[i] is not None else None for i in range(zgnn_len))
                zgnn_kl = tuple(torch.clamp(zgnn_kl[i], 0, 1) if zgnn[i] is not None else None for i in range(zgnn_len))

                # Detach and clamp z:
                zgnn = tuple(zgnn[i].detach() if zgnn[i] is not None else None for i in range(zgnn_len))
                zgnn = tuple(torch.clamp(zgnn[i], 0, 1) if zgnn[i] is not None else None for i in range(zgnn_len))
                if record_interval != -1:
                    zgnn_list.append(deepcopy(tuple(to_np_array(*zgnn, keep_list=True))))
            if "wtarget" in args.ebm_target.split("+"):
                wtarget = wtarget - wtarget_grad * args.step_size_wtarget

                # Update the wtarget_kl:
                wtarget_kl = wtarget_ori
                neg_out_kl_sum = model(img_ori, neg_mask_ori, c_repr=c_repr_ori, z=z_ori, zgnn=zgnn_ori, wtarget=wtarget_kl, batch_shape=batch_shape).sum()
                wtarget_kl_grad = torch.autograd.grad([neg_out_kl_sum], [wtarget_kl], create_graph=True)[0]
                wtarget_kl = wtarget_kl - wtarget_kl_grad * args.step_size_wtarget
                wtarget_kl = torch.clamp(wtarget_kl, 0, 1)

                # Detach and clamp z:
                wtarget = wtarget.detach()
                wtarget = torch.clamp(wtarget, 0, 1)
                if record_interval != -1:
                    wtarget_list.append(deepcopy(to_np_array(wtarget)))

        else:
            # Update subset of {img, neg_mask, c_repr, z} using the gradient:
            if "image" in args.ebm_target.split("+"):
                if args.is_image_tuple:
                    img = tuple(img[i] - img_grad[i] * args.step_size_img for i in range(len(img)))
                    img = tuple(img[i].detach() for i in range(len(img)))
                    img = tuple(torch.clamp(img[i], img_value_min, img_value_max) for i in range(len(img)))
                    if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                        img_list.append(deepcopy(tuple(to_np_array(*img, keep_list=True))))
                else:
                    img = img - img_grad * args.step_size_img
                    img = img.detach()
                    img = torch.clamp(img, img_value_min, img_value_max)
                    if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                        img_list.append(deepcopy(to_np_array(img)))
            if "mask" in args.ebm_target.split("+"):
                neg_mask = tuple(neg_mask[i] - neg_mask_grad[i]*step_size_list[k] for i in range(model.mask_arity))
                neg_mask = tuple(neg_mask[i].detach() for i in range(model.mask_arity))
                neg_mask = tuple(torch.clamp(neg_mask[i], 0, 1) for i in range(model.mask_arity))
                if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                    neg_mask_list.append(deepcopy(tuple(to_np_array(*neg_mask, keep_list=True))))
            if "repr" in args.ebm_target.split("+"):
                c_repr = c_repr - c_repr_grad * args.step_size_repr
                c_repr = c_repr.detach()
                if "softmax" not in args.c_repr_mode:
                    c_repr = torch.clamp(c_repr, 0, 1)
                if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                    c_repr_list.append(deepcopy(to_np_array(c_repr)))
            if "z" in args.ebm_target.split("+"):
                z = tuple(z[i] - z_grad[i] * args.step_size_z if z[i] is not None else None for i in range(z_len))
                z = tuple(z[i].detach() if z[i] is not None else None for i in range(z_len))
                z = tuple(torch.clamp(z[i], 0, 1) if z[i] is not None else None for i in range(z_len))
                if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                    z_list.append(deepcopy(tuple(to_np_array(*z, keep_list=True))))
            if "zgnn" in args.ebm_target.split("+"):
                zgnn = tuple(zgnn[i] - zgnn_grad[i] * args.step_size_zgnn if zgnn[i] is not None else None for i in range(zgnn_len))
                zgnn = tuple(zgnn[i].detach() if zgnn[i] is not None else None for i in range(zgnn_len))
                zgnn = tuple(torch.clamp(zgnn[i], 0, 1) if zgnn[i] is not None else None for i in range(zgnn_len))
                if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                    zgnn_list.append(deepcopy(tuple(to_np_array(*zgnn, keep_list=True))))
            if "wtarget" in args.ebm_target.split("+"):
                wtarget = wtarget - wtarget_grad * args.step_size_wtarget
                wtarget = wtarget.detach()
                wtarget = torch.clamp(wtarget, 0, 1)
                if record_interval != -1 and ((k + 1) % record_interval == 0 or k == args.sample_step - 1):
                    wtarget_list.append(deepcopy(to_np_array(wtarget)))
        neg_out_list.append(deepcopy(to_np_array(neg_out)))

    neg_out_list = np.concatenate(neg_out_list, -1).T   # after: [sample_step, B]
    if record_interval != -1:
        if "image" in args.ebm_target.split("+"):
            if args.is_image_tuple:
                img_list = tuple(Zip(*img_list, function=np.stack))
            else:
                img_list = np.stack(img_list)
        if "mask" in args.ebm_target.split("+"):
            neg_mask_list = tuple(Zip(*neg_mask_list, function=np.stack))
        if "repr" in args.ebm_target.split("+"):
            c_repr_list = np.stack(c_repr_list)
        if "z" in args.ebm_target.split("+"):
            z_list = tuple(Zip(*z_list, function=np.stack))
        if "zgnn" in args.ebm_target.split("+"):
            zgnn_list = tuple(Zip(*zgnn_list))
        if "wtarget" in args.ebm_target.split("+"):
            wtarget_list = np.stack(wtarget)

    info = {
        "neg_out_list": neg_out_list,
        "img_list": img_list,
        "neg_mask_list": neg_mask_list,
        "c_repr_list": c_repr_list,
        "z_list": z_list,
        "zgnn_list": zgnn_list,
        "wtarget_list": wtarget_list,
    }
    if "img_kl" not in locals():
        img_kl = None
    if "neg_mask_kl" not in locals():
        neg_mask_kl = None
    if "c_repr_kl" not in locals():
        c_repr_kl = None
    if "z_kl" not in locals():
        z_kl = None
    if "zgnn_kl" not in locals():
        zgnn_kl = None
    if "wtarget_kl" not in locals():
        wtarget_kl = None
    reg_dict = transform_dict(reg_dict, "array")
    info.update(reg_dict)
    if is_return_E:
        info["E_all"] = E_all
    return (img, neg_mask, c_repr, z, zgnn, wtarget), (img_kl, neg_mask_kl, c_repr_kl, z_kl, zgnn_kl, wtarget_kl), info


# ### 1.3.4 Helper functions for objectives:

# In[ ]:


def get_graph_energy(
    mask,
    mask_info,
    iou_batch_consistency_coef=0,
    iou_concept_repel_coef=0,
    iou_relation_repel_coef=0,
    iou_relation_overlap_coef=0,
    iou_attract_coef=0,
    batch_shape=None,
):
    """
    Obtain the three graph energy that encourages forming a common graph among several examples.

    Args:
        mask: a tuple of masks, each with shape [B, 1, H, W]
        mask_info: a dictionary containing information about the masks. E.g.
            {
                id_to_type: {0: ("concept", 1), 1: ("relation", 0), 2: ("concept", 0), ...},  
                    # The number are chosen from {0,1}, which indicates the number of object slot this mask occupies, for computing mutual_exclusive loss.
                id_same_relation: [(1,3), (4,6), ...],
            }
        batch_shape: if not None, will have shape of [B_task, B_example]
    """
    def get_triu_ids(array, is_triu=True):
        if isinstance(array, Number):
            array = np.arange(array)
        rows_matrix, col_matrix = np.meshgrid(array, array)
        matrix_cat = np.stack([rows_matrix, col_matrix], -1)
        rr, cc = np.triu_indices(len(matrix_cat), k=1)
        rows, cols = matrix_cat[cc, rr].T
        return rows, cols

    def get_relation_overlap_loss_pair(distance_matrix_batch, rel_tuple_1, rel_tuple_2):
        """distance_matrix_batch: [B_task, B_example, n_masks, n_masks]"""
        shape = distance_matrix_batch.shape
        assert len(shape) == 4
        assert shape[2] == shape[3]
        rel_tuple_2_r = (rel_tuple_2[1], rel_tuple_2[0])
        loss = (1 - distance_matrix_batch[:, :, rel_tuple_1, rel_tuple_2]).clamp(0, 1).prod(-1) +                (1 - distance_matrix_batch[:, :, rel_tuple_1, rel_tuple_2_r]).clamp(0, 1).prod(-1)
        return loss

    assert isinstance(mask, tuple) or isinstance(mask, list)
    assert mask[0].shape[1] == 1
    n_masks = len(mask)
    device = mask[0].device
    batch_size = mask[0].shape[0]
    loss_dict = {}

    # Get pairwise Jaccard distance between masks:
    mask = torch.stack(mask, 1)  # [B, n_masks, 1, H, W]
    if batch_shape is None:
        mask = mask[None]  # [B_task:1, B_example, n_masks, 1, H, W]
        batch_shape = (1, batch_size)
    else:
        mask = mask.view(*batch_shape, *mask.shape[1:])  # [B_task, B_example, n_masks, 1, H, W]
    mask_expand_0 = mask[:,:,:,None]
    mask_expand_1 = mask[:,:,None]
    distance_matrix_batch = get_soft_Jaccard_distance(mask_expand_0, mask_expand_1, dim=(-3,-2,-1))  # [B_task, B_example, n_masks, n_masks]
    distance_matrix_mean = distance_matrix_batch.mean(1, keepdims=True)  # [B_task, 1, n_masks, n_masks]

    # Encourage that the pairs in the same batch has similar distances across examples:
    loss_all = torch.FloatTensor([[[0]]]).to(device)
    if iou_batch_consistency_coef > 0 and batch_size > 1:
        loss_batch_consistency = ((distance_matrix_batch - distance_matrix_mean).square().mean(1) + 1e-10).sqrt().mean((1,2), keepdims=True).expand(-1, batch_shape[1], 1) * iou_batch_consistency_coef  # [B_task, B_example, 1]
        loss_all = loss_all + loss_batch_consistency
        loss_dict["loss_batch_consistency"] = to_np_array(loss_batch_consistency)

    # Encourage that the masks for different concepts do not overlap:
    repel_rows = []
    repel_cols = []
    concept_ids = [id for id, item in mask_info["id_to_type"].items() if item[0] == "concept"]
    concept_rows, concept_cols = get_triu_ids(concept_ids)
    repel_rows.append(concept_rows)
    repel_cols.append(concept_cols)
    if iou_concept_repel_coef > 0:
        loss_concept_repel = (1 - distance_matrix_batch[:,:,concept_rows,concept_cols].mean(-1, keepdims=True)) * iou_concept_repel_coef  # [B_task, B_example, 1]
        loss_all = loss_all + loss_concept_repel
        loss_dict["loss_concept_repel"] = to_np_array(loss_concept_repel)

    # Encourage the mask for the same relation do not overlap:
    if len(mask_info["id_same_relation"]) > 0:
        relation_rows, relation_cols = np.stack(mask_info["id_same_relation"]).T
    else:
        relation_rows, relation_cols = np.array([]), np.array([])
    repel_rows.append(relation_rows)
    repel_cols.append(relation_cols)
    if iou_relation_repel_coef > 0 and len(mask_info["id_same_relation"]) > 0:
        loss_relation_repel = (1 - distance_matrix_batch[:,:,relation_rows,relation_cols].mean(-1, keepdims=True)) * iou_relation_repel_coef  # [B_task, B_example, 1]
        loss_all = loss_all + loss_relation_repel
        loss_dict["loss_relation_repel"] = to_np_array(loss_relation_repel)

    # Discourage two relation-EBMs to discover the same pair of objects:
    if iou_relation_overlap_coef > 0 and len(mask_info["id_same_relation"]) > 1:
        id_same_relation = mask_info["id_same_relation"]
        length = len(id_same_relation)
        pairs = [(id_same_relation[i], id_same_relation[j]) for i in range(length) for j in range(length) if i < j]
        loss_list = [get_relation_overlap_loss_pair(distance_matrix_batch, rel_tuple_1, rel_tuple_2) for rel_tuple_1, rel_tuple_2 in pairs]
        loss_relation_overlap = torch.stack(loss_list, -1).sum(-1, keepdims=True) * iou_relation_overlap_coef
        loss_all = loss_all + loss_relation_overlap
        loss_dict["loss_relation_overlap"] = to_np_array(loss_relation_overlap)

    # Encourage that compatible (not repelled) masks can snap to each other when they are near enough:
    if iou_attract_coef > 0:
        all_rows, all_cols = get_triu_ids(n_masks)
        repel_rows = np.concatenate(repel_rows)
        repel_cols = np.concatenate(repel_cols)
        all_tuples = [(row, col) for row, col in zip(all_rows, all_cols)]
        repel_tuples = [(row, col) for row, col in zip(repel_rows, repel_cols)]
        compat_tuples = [ele for ele in all_tuples if ele not in repel_tuples]
        if len(compat_tuples) > 0:
            compat_rows, compat_cols = np.array(compat_tuples).T
            distance_attract = distance_matrix_mean[:,:,compat_rows, compat_cols].clamp(1e-5)  # [B_task, 1, n_compat]
            print("\nDistance_attract:")
            print(distance_attract)
            loss_attract = distance_attract.mean(-1, keepdims=True).expand(-1, batch_shape[1], 1) * iou_attract_coef  # [B_task, B_example, 1]
            loss_all = loss_all + loss_attract
            loss_dict["loss_attract"] = to_np_array(loss_attract)
    return loss_all, loss_dict


def get_pixel_gm(mask_list, order=-1, epsilon=1e-5):
    """Get generalized-mean over the distance to 0 and 1 over multiple masks for each pixel.
        Use an order <=0 or "min" to encourge specialization of masks.
    """
    assert isinstance(mask_list, list) or isinstance(mask_list, tuple)
    shape = mask_list[0].shape
    assert len(shape) == 4 and shape[1] == 1
    n_masks = len(mask_list)

    mask_list = torch.stack(mask_list, -1)  # Last dimension is the n_masks dimension
    if isinstance(order, Number):
        mask_list = mask_list.clamp(epsilon, 1-epsilon)

    if order == -1:
        L1 = (n_masks / (1 / mask_list).sum(-1)).mean((1,2,3))
        L2 = (n_masks / (1 / (1-mask_list)).sum(-1)).mean((1,2,3))
    elif order == 0:
        L1 = (mask_list.prod(-1) ** (1/n_masks)).mean((1,2,3))
        L2 = ((1-mask_list).prod(-1) ** (1/n_masks)).mean((1,2,3))
    elif order == 1:
        L1 = mask_list.mean((1,2,3,4))
        L2 = (1-mask_list).mean((1,2,3,4))
    elif order == "max":
        L1 = mask_list.max(4)[0].mean((1,2,3))
        L2 = (1-mask_list).max(-1)[0].mean((1,2,3))
    elif order == "min":
        L1 = mask_list.min(4)[0].mean((1,2,3))
        L2 = (1-mask_list).min(-1)[0].mean((1,2,3))
    else:
        assert isinstance(order, Number)
        L1 = (((mask_list ** order).mean(-1)) ** (1 / float(order))).mean((1,2,3))
        L2 = ((((1-mask_list) ** order).mean(-1)) ** (1 / float(order))).mean((1,2,3))
    pixel_gm = L1 + L2
    return pixel_gm


def get_pixel_entropy(mask_list, epsilon=1e-5):
    """Obtain the pixel-wise entropy for each mask.
    Args:
        mask_list: each mask in mask_list should have the shape of [B, 1, H, W]
        epsilon: to prevent NaN.

    Returns:
        entropy: shape [B,]
    """
    # Make sure that the format is correct:
    assert isinstance(mask_list, list) or isinstance(mask_list, tuple)
    shape = mask_list[0].shape
    assert len(shape) == 4 and shape[1] == 1
    
    mask_list = torch.stack(mask_list, 1).clamp(epsilon, 1-epsilon)
    entropy = (-mask_list * torch.log(mask_list) - (1-mask_list) * torch.log(1-mask_list)).mean((1,2,3,4))
    return entropy


def get_mask_entropy(mask_list, threshold=0.01, epsilon=1e-5):
    """Obtain the pixel-wise entropy for each mask.
    Args:
        mask_list: each mask in mask_list should have the shape of [B, 1, H, W]
        epsilon: to prevent NaN.

    Returns:
        entropy: shape [B,]
    """
    def get_entropy(values):
        return (-values * torch.log(values) - (1-values) * torch.log(1-values))
    # Make sure that the format is correct:
    assert isinstance(mask_list, list) or isinstance(mask_list, tuple)
    shape = mask_list[0].shape
    assert len(shape) == 4 and shape[1] == 1
    
    mask_list = torch.stack(mask_list, 1)
    # Take mean over channel, height, and width
    thresholded = torch.where(mask_list > threshold, 1, 0)
    total_pixels = thresholded.sum((-3, -2, -1))
    overall_values = torch.zeros(total_pixels.shape).to(total_pixels.device)
    # Take mean over C, H, W
    overall_values =  torch.where(total_pixels > 0, (thresholded * mask_list).sum((-3, -2, -1)) / total_pixels, overall_values)
    entropy = get_entropy(overall_values.clamp(epsilon, 1-epsilon)).mean(1)
    return entropy


def get_neg_mask_overlap(
    neg_mask: tuple,
    mask_info=None,
    is_penalize_lower="True",
    img=None,
):
    """
    Calculates penalty energy from negative masks overlapping with each other.

    Args:
        neg_mask: tuple of masks, each of shape [ensemble size, 1, H, W]
        is_penalize_lower: if True or "True", will penalize that the sum is less than 1.
            If "False" or False, will not. If "obj:0.1", will only penalize on the object locations (if n_channels==10), times coefficient of 0.1.
        img: image, required if is_penalize_lower == "obj".

    Returns:
        tensor of shape [N,] representing each mask's energy
    """
    # Stack all the masks on top of each other and sum over them
    if mask_info is None or "id_to_type" not in mask_info:
        neg_mask_obj = neg_mask
    else:
        neg_mask_obj = [neg_mask[i] for i in range(len(neg_mask)) if mask_info["id_to_type"][i][1] == 1]
    neg_mask_overlap = torch.cat(neg_mask_obj, 1).sum(1, keepdims=True)
    if mask_info is not None and "mask_exclude" in mask_info:
        neg_mask_overlap = neg_mask_overlap + mask_info["mask_exclude"]
    # Penalize the distance of the sum of masks compared to 1:
    if is_penalize_lower == "True" or is_penalize_lower is True:
        neg_mask_overlap = (neg_mask_overlap - 1).abs()
    elif is_penalize_lower == "False" or is_penalize_lower is False:
        neg_mask_overlap = (neg_mask_overlap - 1).clamp(0)
    elif is_penalize_lower.startswith("obj"):
        penalize_lower_coef = eval(is_penalize_lower.split(":")[1]) if len(is_penalize_lower.split(":")) == 2 else 1
        neg_mask_overlap_exceed = (neg_mask_overlap - 1).clamp(0)
        assert len(img.shape) == 4
        if img.shape[1] == 10:
            obj_mask = (img[:, :1] != 1).float()
            neg_mask_overlap_under = (obj_mask - neg_mask_overlap).clamp(0)
            neg_mask_overlap = neg_mask_overlap_exceed + neg_mask_overlap_under * penalize_lower_coef
        else:
            neg_mask_overlap = neg_mask_overlap_exceed
    else:
        raise Exception("is_penalize_lower '{}' is not valid!".format(is_penalize_lower))
    return neg_mask_overlap.mean((1, 2, 3))


def get_fine_neg_mask_overlap(neg_mask: tuple, threshold=0.01, epsilon=1e-5, mask_info=None, is_penalize_lower=True):
    """
    Calculates penalty energy from negative masks overlapping with each other. Penalizes
    overlap where sum of masks may not be greater than 1

    Args:
        neg_mask: tuple of masks, each of shape [ensemble size, 1, H, W]
        is_penalize_lower: if True, will penalize that the sum is less than 1.

    Returns:
        tensor of shape [N,] representing each mask's energy
    """
    # Stack all the masks on top of each other and sum over them
    if mask_info is None:
        neg_mask_obj = neg_mask
    else:
        neg_mask_obj = [neg_mask[i] for i in range(len(neg_mask)) if mask_info["id_to_type"][i][1] == 1]
    neg_mask_obj = torch.cat(neg_mask_obj, 1)
    # Mask out pixels that are less than the threshold 
    neg_mask_thresh = (neg_mask_obj > threshold) * neg_mask_obj
    neg_mask_overlap = neg_mask_thresh.sum(1, keepdims=True)
    # Expand the sum of masks dimension
    expanded_overlap = neg_mask_overlap.expand(-1, neg_mask_thresh.shape[1], -1, -1)
    other_mask = expanded_overlap - neg_mask_thresh
    
    overlapping = torch.where((other_mask > threshold).logical_and(neg_mask_thresh > threshold), 1.0, 0.0)
    mean_mask = neg_mask_thresh.mean((2, 3)) # Take mean over H, W
    mean_overlap = (neg_mask_thresh * overlapping).mean((2, 3))
    normalized_overlap = mean_overlap / mean_mask.clamp(min=epsilon)
    assert torch.all(normalized_overlap <= 1.0)
    mean_overlap = (normalized_overlap).mean(1) # Take mean over EBM's
    return mean_overlap


def get_neg_mask_exceed(neg_mask: tuple, pos_all_mask: Union[torch.Tensor, tuple]):
    """
    Calculates penalty energy from a negative mask appearing where there is no
    object in reality (aka, whenever negative mask isn't a subset of ground
    truth mask).

    :param neg_mask: tuple of masks, each of shape [N, 1, H, W]
    :param pos_all_mask: ground truth mask(s), where each [H, W] contains the
    mask for *all* objects in the image. Either tensor of shape [N, 1, H, W] or
    tuple. If tuple, it doesn't represent all the objects?
    :returns: tensor of shape [N,] representing each mask's energy
    """

    # TODO pos_all_mask tuple
    neg_mask_stack = torch.cat(neg_mask, 1).sum(1, keepdims=True).clamp(max=1)

    exceeding_regions = neg_mask_stack * pos_all_mask.logical_not()

    return exceeding_regions.mean((1, 2, 3))


def get_neg_mask_exceed_energy(neg_mask, pos_all_mask):
    """
    Calculates "penalty" energy given a model-predicted negative mask
    and a positive ground-truth mask, which increases when:

    1. Two negative masks overlap
    2. A negative mask appears in a place where there is no object in reality.
    """

    if isinstance(pos_all_mask, tuple):
        assert len(neg_mask) == 2
        neg_mask_exceed = torch.cat([torch.clamp(neg_mask[0] - pos_all_mask[0], min=0),
                                     torch.clamp(neg_mask[1] - pos_all_mask[1], min=0)], 1).sum(1, keepdims=True)
    else:
        neg_mask_sum = torch.cat(neg_mask, 1).sum(1, keepdims=True)
        # Calculate number of times mask exceeds boundary of positive ground truth
        neg_mask_exceed = torch.clamp(neg_mask_sum - pos_all_mask, min=0)
    neg_mask_exceed_energy = neg_mask_exceed.mean((1,2,3))
    return neg_mask_exceed_energy


# # 2. GNN:

# ## 2.1 GNLayer:

# In[ ]:


class GNLayer(MessagePassing):
    """One GN layer of edge update followed by node update."""
    def __init__(
        self,
        input_size,
        edge_attr_size,
        n_neurons,
        mlp_n_layers,
        aggr_mode="max",
        output_size=None,
        activation="relu",
        is_output_edge_feature=False,
    ):
        """
        Initialization for the GNLayer.
        Args:
            input_size: number of features for each node.
            edge_attr_size: number of features for edge attribute.
            n_neurons: number of neurons for each layer of the node_mlp and edge_mlp.
            mlp_n_layers: number of layers for the node_mlp and edge_mlp.
            output_size: output size. If None, will use edge_attr_size.
            activation: activation for the hidden layers for the node_mlp and edge_mlp.
        """
        super(GNLayer, self).__init__(aggr=aggr_mode)
        self.is_output_edge_feature = is_output_edge_feature
        self.node_mlp = MLP(input_size=n_neurons + input_size,
                            n_neurons=n_neurons,
                            n_layers=mlp_n_layers,
                            output_size=edge_attr_size if output_size is None else output_size,
                            activation=activation,
                           )
        self.edge_mlp = MLP(input_size=2 * input_size + edge_attr_size,
                            n_neurons=n_neurons,
                            n_layers=mlp_n_layers,
                            activation=activation,
                           )

    def forward(self, x, edge_index, edge_attr):
        """Main forward function"""
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Computes the aggregated results after summing over messages from neighboring edges for each node:
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
        # Use an MLP on the concatenation of original node features and the aggregated messages:
        out = self.node_mlp(torch.cat([x, out], 1))
        if self.is_output_edge_feature:
            return out, self.edge_feature
        else:
            return out

    def message(self, x_i, x_j, edge_attr):
        """Computes the message given features of the source node (x_i) and features of the target node."""
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_attr has shape [E, in_channels]
        out = torch.cat([x_i, x_j, edge_attr], dim=1)  # [E, 2 * in_channels + edge_attr_size]
        out = self.edge_mlp(out)
        if self.is_output_edge_feature:
            self.edge_feature = out
        return out

    def update(self, aggr_out):
        """The update function, after summing over the messages from the neighboring edges."""
        # aggr_out has shape [N, out_channels]
        return aggr_out


# ## 2.2 GNNs:

# ### 2.2.1 GNN2:

# In[ ]:


class GNN2(nn.Module):
    def __init__(
        self,
        input_size,
        edge_attr_size,
        n_GN_layers,
        n_neurons=32,
        GNN_output_size=8,
        mlp_n_layers=2,
        normalize="None",
        activation="relu",
        recurrent=False,
        aggr_mode="max",
        is_output_edge_feature=False,
    ):
        """
        Args:
            input_size: number of features for each node.
            edge_attr_size: number of features for edge attribute.
            n_GN_layers: number of GNLayers.
            n_neurons: number of neurons for each layer of the node_mlp and edge_mlp.
            GNN_output_size: output_size of each GNLayer. If None, will use edge_attr_size.
            mlp_n_layers: number of layers for the node_mlp and edge_mlp.
            normalize: normalization mode. Choose from "None", "layer", "batch".
            activation: activation for the hidden layers for the node_mlp and edge_mlp.
        """
        super(GNN2, self).__init__()
        self.input_size = input_size
        self.n_GN_layers = n_GN_layers
        self.GNN_output_size = GNN_output_size
        self.normalize = normalize
        self.aggr_mode = aggr_mode
        self.is_output_edge_feature = is_output_edge_feature
        for i in range(1, self.n_GN_layers + 1):
            setattr(self, "layer_{}".format(i),
                    GNLayer(
                    input_size=input_size if i == 1 else GNN_output_size,
                    edge_attr_size=edge_attr_size,
                    aggr_mode=aggr_mode,
                    n_neurons=n_neurons,
                    mlp_n_layers=mlp_n_layers,
                    output_size=GNN_output_size,
                    activation=activation,
                    is_output_edge_feature=is_output_edge_feature if i == self.n_GN_layers else False,
            ))
            if i != self.n_GN_layers:
                if self.normalize == "layer":
                    setattr(self, "norm_{}".format(i), nn.LayerNorm(GNN_output_size))
                elif self.normalize.startswith("gn"):
                    n_groups = eval(self.normalize.split("-")[1])
                    setattr(self, "norm_{}".format(i), nn.GroupNorm(n_groups, GNN_output_size, affine=True))
                elif self.normalize == "batch":
                    setattr(self, "norm_{}".format(i), nn.BatchNorm1d(GNN_output_size))
                elif self.normalize == "None":
                    pass
                else:
                    raise Exception("Normalize '{}' is not valid!".format(self.normalize))
        self.rnn = None 
        if recurrent:
            # num input features, num hidden features. Always uses bias by default.
            # Batch size is the number of nodes in the graph
            self.rnn = nn.GRUCell(GNN_output_size, GNN_output_size)

    def forward(
        self,
        data,
        rnn_hx=None,
    ):
        x = data.x
        for i in range(1, self.n_GN_layers + 1):
            if i == self.n_GN_layers and self.is_output_edge_feature:
                x, edge_features = getattr(self, "layer_{}".format(i))(x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            else:
                x = getattr(self, "layer_{}".format(i))(x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            if i != self.n_GN_layers and (self.normalize in ["layer", "batch"] or self.normalize.startswith("gn")):
                x = getattr(self, "norm_{}".format(i))(x)

        embed_hx_new = None
        if self.rnn is not None:
            # Pass in input, hidden state
            x = self.rnn(x, rnn_hx)
            embed_hx_new = x.clone()
        if self.is_output_edge_feature:
            return (x, edge_features), embed_hx_new
        else:
            return x, embed_hx_new


# ### 2.2.2 GNN_energy:

# In[ ]:


class GNN_energy(nn.Module):
    def __init__(
        self,
        mode,
        is_zgnn_node,
        edge_attr_size,
        aggr_mode,
        n_GN_layers,
        n_neurons=32,
        GNN_output_size=8,
        mlp_n_layers=2,
        gnn_normalization_type="None",
        activation="relu",
        recurrent=False,
        cnn_output_size=8,
        cnn_is_spec_norm=True,
        cnn_normalization_type="None",
        cnn_channel_base=64,
        cnn_aggr_mode="sum",
        c_repr_dim=8,
        z_dim=8,
        zgnn_dim=1,
        distance_loss_type="Jaccard",
        pooling_type="gated",
        pooling_dim=32,
        is_x=False,
    ):
        """Combine multiple modalities (w, z, c, zgnn, wtarget) together via CNN and GNN, 
        and return a single energy.
        
        Args:
            mode: choose from "concat", "softmax"
        """
        super().__init__()
        self.mode = mode
        self.is_zgnn_node = is_zgnn_node
        self.is_x = is_x
        if self.mode == "softmax":
            assert zgnn_dim == 1
            self.softmax_coef = nn.Parameter(torch.ones(1) * 0.2)
            self.in_channels = 1
        else:
            self.in_channels = 1
        self.gnn_input_size = cnn_output_size + z_dim + c_repr_dim
        if self.is_zgnn_node:
            self.gnn_input_size += zgnn_dim
            self.in_channels += 1
        else:
            if self.is_x:
                self.in_channels = 10
        self.cnn_aggr_mode = cnn_aggr_mode
        self.edge_attr_size = edge_attr_size
        self.aggr_mode = aggr_mode
        self.n_GN_layers = n_GN_layers
        self.n_neurons = n_neurons
        self.GNN_output_size = GNN_output_size
        self.mlp_n_layers = mlp_n_layers
        self.gnn_normalization_type = gnn_normalization_type
        self.activation = activation
        self.recurrent = recurrent
        self.cnn_output_size = cnn_output_size
        self.cnn_is_spec_norm = cnn_is_spec_norm
        self.cnn_normalization_type = cnn_normalization_type
        self.cnn_channel_base = cnn_channel_base
        self.cnn_aggr_mode = cnn_aggr_mode
        self.c_repr_dim = c_repr_dim
        self.z_dim = z_dim
        self.zgnn_dim = zgnn_dim
        self.distance_loss_type = distance_loss_type
        self.pooling_type = pooling_type
        self.pooling_dim = pooling_dim

        if distance_loss_type == "mse":
            self.loss_fun = lambda x, y: nn.MSELoss(reduction="none")(x, y).mean((-3,-2,-1))
        elif distance_loss_type == "Jaccard":
            self.loss_fun = lambda x, y: get_soft_Jaccard_distance(x, y, dim=(-3,-2,-1))
        else:
            raise
        self.gnn = GNN2(
            input_size=self.gnn_input_size,
            edge_attr_size=edge_attr_size,
            aggr_mode=aggr_mode,
            n_GN_layers=n_GN_layers,
            n_neurons=n_neurons,
            GNN_output_size=GNN_output_size,
            mlp_n_layers=mlp_n_layers,
            normalize=gnn_normalization_type,
            activation=activation,
            recurrent=recurrent,
            is_output_edge_feature=True if pooling_type == "gated" else False,
        )
        self.cnn = nn.Sequential(
            CResBlock(self.in_channels, cnn_channel_base, downsample=True, is_spec_norm=cnn_is_spec_norm, c_repr_mode="None", z_mode="None", act_name=activation, normalization_type=cnn_normalization_type),
            CResBlock(cnn_channel_base, cnn_channel_base, is_spec_norm=cnn_is_spec_norm, c_repr_mode="None", z_mode="None", act_name=activation, normalization_type=cnn_normalization_type),
            CResBlock(cnn_channel_base, cnn_channel_base*2, downsample=True, is_spec_norm=cnn_is_spec_norm, c_repr_mode="None", z_mode="None", act_name=activation, normalization_type=cnn_normalization_type),
            CResBlock(cnn_channel_base*2, cnn_channel_base*2, is_spec_norm=cnn_is_spec_norm, c_repr_mode="None", z_mode="None", act_name=activation, normalization_type=cnn_normalization_type),
            CResBlock(cnn_channel_base*2, cnn_channel_base*2, downsample=True, is_spec_norm=cnn_is_spec_norm, c_repr_mode="None", z_mode="None", act_name=activation, normalization_type=cnn_normalization_type),
            CResBlock(cnn_channel_base*2, cnn_channel_base*2, is_spec_norm=cnn_is_spec_norm, c_repr_mode="None", z_mode="None", act_name=activation, normalization_type=cnn_normalization_type),
        )
        self.cnn_mlp = MLP(
            cnn_channel_base*2,
            n_neurons=cnn_channel_base,
            n_layers=1,
            activation=activation,
            output_size=cnn_output_size,
        )
        if self.z_dim > 0:
            self.g_z = MLP(
                z_dim,
                n_neurons=z_dim,
                n_layers=1,
                activation=activation,
                output_size=z_dim,
            )
        if self.pooling_type == "mean":
            self.gnn_mlp = MLP(
                GNN_output_size,
                n_neurons=GNN_output_size,
                n_layers=1,
                activation="relu",
                output_size=1,
            )
        elif self.pooling_type == "gated":
            self.node_attn = nn.Linear(GNN_output_size, 1)
            self.node_pooling = nn.Linear(GNN_output_size, pooling_dim)
            self.edge_attn = nn.Linear(n_neurons, 1)
            self.edge_pooling = nn.Linear(n_neurons, pooling_dim)
            self.gnn_mlp = MLP(
                pooling_dim * 2,
                n_neurons=pooling_dim,
                n_layers=2,
                activation="relu",
                last_layer_linear=False,
                output_size=1,
            )
        else:
            raise


    def cnn_forward(self, input):
        out = self.cnn(input)
        if self.cnn_aggr_mode == "sum":
            out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        elif self.cnn_aggr_mode == "max":
            out = out.view(out.shape[0], out.shape[1], -1).max(2)[0]
        elif self.cnn_aggr_mode == "mean":
            out = out.view(out.shape[0], out.shape[1], -1).mean(2)
        else:
            raise
        out = self.cnn_mlp(out)
        return out


    def get_pyg_data(self, w, z, c, zgnn, wtarget, batch_shape, x=None):
        def get_edge_index_aug(n_ebms, batch_size):
            edge_matrix = np.stack(np.meshgrid(np.arange(n_ebms), np.arange(n_ebms)), -1).reshape(-1,2)
            mask = edge_matrix[:,0] != edge_matrix[:,1]
            edge_index = torch.LongTensor(edge_matrix[mask].T)
            edge_index_aug = edge_index[:,None]
            edge_index_aug = edge_index_aug + (torch.arange(batch_size) * n_ebms)[None,:,None]
            edge_index_aug = edge_index_aug.reshape(2, -1)
            return edge_index_aug

        n_ebms = len(w)
        batch_size_all = np.prod(batch_shape)
        device = w[0].device
        if self.mode == "concat":
            w_embed = torch.stack([self.cnn_forward(torch.cat([w_ele, wtarget], 1) if self.is_zgnn_node else w_ele * x if self.is_x and x is not None else w_ele) for w_ele in w], 1)  # each has shape [B_task * B_example, n_ebms, cnn_output_size]
        elif self.mode == "softmax":
            w_embed = torch.stack([self.cnn_forward(w_ele) for w_ele in w], 1)  # each has shape [B_task * B_example, n_ebms, cnn_output_size]
        else:
            raise Exception("mode '{}' is not valid!".format(self.mode))
        c_embed = torch.stack(c, 1)  # [1, n_ebms, c_repr_dim]
        c_embed = c_embed.expand(w_embed.shape[0], n_ebms, c_embed.shape[-1])     # [B_task * B_example, n_ebms, c_repr_dim]

        if self.z_dim > 0:
            z_embed = self.g_z(torch.stack(z, 1))  # [B_task * B_example, n_ebms, z_dim]
            all_embed = [w_embed, z_embed, c_embed]
        else:
            all_embed = [w_embed, c_embed]
        if self.is_zgnn_node:
            zgnn_node = zgnn[0][:,None].expand(*batch_shape, *zgnn[0].shape[-2:])  # [B_task, B_example, n_ebms, Z_node_size]
            zgnn_node_embed = zgnn_node.reshape(-1, *zgnn_node.shape[-2:])         # [B_task * B_example, n_ebms, Z_node_size]
            all_embed.append(zgnn_node_embed)
        data_x = torch.cat(all_embed, -1)   # [B_task * B_example, n_ebms, gnn_input_size]
        data_edge_index = get_edge_index_aug(n_ebms, batch_size_all).to(device)    # [2, B_task * B_example * n_edges], n_edges = n_ebm * (n_ebm-1)
        zgnn_edge_embed = zgnn[1][:,None].expand(*batch_shape, *zgnn[1].shape[-2:])# zgnn[1]: [B_task, n_edges, edge_attr_size], zgnn_edge_embed: [B_task, B_example, n_edges, edge_attr_size]
        data_edge_attr = zgnn_edge_embed.reshape(-1, zgnn_edge_embed.shape[-1])    # [B_task * B_example * n_edges, edge_attr_size]

        data = Data(x=data_x.view(-1, data_x.shape[-1]), edge_index=data_edge_index, edge_attr=data_edge_attr)
        return data


    def forward(self, w, z, c, zgnn, wtarget, batch_shape, x=None):
        data = self.get_pyg_data(w, z, c, zgnn, wtarget, batch_shape=batch_shape, x=x)
        assert data.x.shape[-1] == self.gnn.input_size

        if self.pooling_type == "mean":
            gnn_out, _ = self.gnn(data)
            gnn_out = self.gnn_mlp(gnn_out).view(*batch_shape, -1)
            E_all = gnn_out.mean(-1, keepdims=True)  # [B_task, B_example, 1]
        elif self.pooling_type == "gated":
            n_nodes = len(w)
            n_edges = n_nodes * (n_nodes-1)
            (node_feature_out, edge_feature_out), _ = self.gnn(data)
            node_feature_out = node_feature_out.view(*batch_shape, n_nodes, -1)  # [B_task, B_example, n_nodes, GNN_output_size]
            edge_feature_out = edge_feature_out.view(*batch_shape, n_edges, -1)  # [B_task, B_example, n_nodes, n_neurons]

            node_pool = self.node_pooling(node_feature_out * self.node_attn(node_feature_out)).sum(-2) # [B_task, B_example, pool_dim]
            edge_pool = self.edge_pooling(edge_feature_out * self.edge_attn(edge_feature_out)).sum(-2) # [B_task, B_example, pool_dim]
            pool = torch.cat([node_pool, edge_pool], -1)
            E_all = self.gnn_mlp(pool)
        if self.mode == "softmax":
            prob = zgnn[0] / (zgnn[0].abs().sum(1, keepdims=True) + 1e-5)  # [B_task, n_ebms, 1]
            entropy = - (prob * torch.log(prob.clamp(1e-5))).sum(1, keepdims=True)
            w_reshape = torch.stack([w_ele.view(*batch_shape, *w_ele.shape[-3:]) for w_ele in w], 1)  # [B_task, n_ebms, B_example, 1, H, W]
            w_weighted = (w_reshape * prob[...,None,None,None]).sum(1)  # 
            w_weighted = w_weighted.view(-1, *w_weighted.shape[-3:])
            E_fit = self.loss_fun(w_weighted, wtarget).view(*batch_shape, 1) * self.softmax_coef
            entropy_expand = entropy.expand(E_fit.shape) * self.softmax_coef / 3
            E_all = E_all + E_fit
        return E_all

    @property
    def model_dict(self):
        model_dict = {"type": "GNN_energy"}
        model_dict["mode"] = self.mode
        model_dict["is_zgnn_node"] = self.is_zgnn_node
        model_dict["edge_attr_size"] = self.edge_attr_size
        model_dict["aggr_mode"] = self.aggr_mode
        model_dict["n_GN_layers"] = self.n_GN_layers
        model_dict["n_neurons"] = self.n_neurons
        model_dict["GNN_output_size"] = self.GNN_output_size
        model_dict["mlp_n_layers"] = self.mlp_n_layers
        model_dict["gnn_normalization_type"] = self.gnn_normalization_type
        model_dict["activation"] = self.activation
        model_dict["recurrent"] = self.recurrent
        model_dict["cnn_output_size"] = self.cnn_output_size
        model_dict["cnn_is_spec_norm"] = self.cnn_is_spec_norm
        model_dict["cnn_normalization_type"] = self.cnn_normalization_type
        model_dict["cnn_channel_base"] = self.cnn_channel_base
        model_dict["cnn_aggr_mode"] = self.cnn_aggr_mode
        model_dict["c_repr_dim"] = self.c_repr_dim
        model_dict["z_dim"] = self.z_dim
        model_dict["zgnn_dim"] = self.zgnn_dim
        model_dict["distance_loss_type"] = self.distance_loss_type
        model_dict["pooling_type"] = self.pooling_type
        model_dict["pooling_dim"] = self.pooling_dim
        model_dict["is_x"] = self.is_x
        model_dict["state_dict"] = to_cpu(self.state_dict())
        return model_dict


# In[ ]:


if __name__ == "__main__":
    device = "cuda:0"
    batch_shape = (20,6)
    w = (torch.rand(120,1,8,8), torch.rand(120,1,8,8), torch.rand(120,1,8,8), torch.rand(120,1,8,8))
    z = (torch.rand(120,5),torch.rand(120,5),torch.rand(120,5),torch.rand(120,5))
    c = (torch.rand(1,8), torch.rand(1,8), torch.rand(1,8), torch.rand(1,8))
    zgnn = (torch.rand(20,4,1), torch.rand(20,12,8))
    wtarget = None #torch.rand(12,1,8,8)
    w, z, c, zgnn, wtarget = to_device_recur((w, z, c, zgnn, wtarget), device=device)

    self = GNN_energy(
        mode="concat",
        is_zgnn_node=False,
        aggr_mode="add",
        edge_attr_size=8,
        n_GN_layers=2,
        z_dim=5,
        zgnn_dim=1,
        distance_loss_type="Jaccard",
        pooling_type="gated",
        pooling_dim=32,
        cnn_is_spec_norm=True,
    ).to(device)
    out = self(w, z, c, zgnn, wtarget, batch_shape)
    self.model_dict


# # 3. LambdaNet

# ## 3.1 LambdaLayer

# In[ ]:


# MIT License
# Adapted from https://github.com/lucidrains/lambda-networks

from torch import einsum
from einops import rearrange

# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

# lambda layer

class LambdaLayer(nn.Module):
    def __init__(
        self,
        # Dimension of the query
        dim_q,
        # Dimension of the context
        dim_c,
        *,
        dim_k,
        # The length of the context
        m = None,
        r = None,
        heads = 4,
        # Dimension of the value if heads=1
        dim_out = None,
        dim_u = 1,
        use_relpos=True):
        super().__init__()
        dim_out = default(dim_out, dim_c)
        self.u = dim_u # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads
        self.to_q = nn.Conv2d(dim_q, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim_c, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim_c, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = exists(r)
        self.use_relpos = use_relpos
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        elif self.use_relpos:
            assert exists(m), 'You must specify the window size (m=h=w)'
            rel_lengths = 2 * m - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
            self.rel_pos = calc_rel_pos(m)

    def forward(self, x, context):
        """Take in both x and the context."""
        b, c, hh, ww, u, h = *context.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = k.softmax(dim=-1)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        if self.local_contexts:
            v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = self.pos_conv(v)
            Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
        elif self.use_relpos:
            n, m = self.rel_pos.unbind(dim = -1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b h v n', q, λp)
        else:
            # Use the shape of the queries
            _, _, hh, ww = x.shape
            Yp = torch.zeros(Yc.shape, device=Yc.device)

        Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out


# ## 3.2 Test LambdaLayer

# In[ ]:


if __name__ == "__main__":
    # Batch size, num_channels, height, width
    x = torch.randn(1, 32, 64, 64)
    c = torch.clone(x)
    layer = LambdaLayer(
        dim_q = 32,
        dim_c = 32,       # channels going in
        dim_out = 32,   # channels out
        m = 64,         # size of the receptive window - max(height, width)
        dim_k = 16,     # key dimension
        heads = 4,      # number of heads, for multi-query
        dim_u = 1       # 'intra-depth' dimension
    )
    out = layer(x, c)
    print(out.size())


# In[ ]:


if __name__ == "__main__":
    # Test query having different dimension from context, as well as different
    # height and width
    x = torch.randn(1, 10, 1, 50)
    c = torch.randn(1, 32, 64, 64)
    layer = LambdaLayer(
        dim_q = 10,
        dim_c = 32,       # channels going in
        dim_out = 32,   # channels out
        m = 64,         # size of the receptive window - max(height, width)
        dim_k = 16,     # key dimension
        heads = 4,      # number of heads, for multi-query
        dim_u = 1,       # 'intra-depth' dimension
        use_relpos=False # Using relpos embedding when query isn't same shape as 
                        # context isn't implemented
    )
    out = layer(x, c)
    print(out.size())


# # 4. MONet:

# ## 4.1 AttentionNet:

# In[ ]:


# License: MIT
# Adapted from https://github.com/stelzner/monet

def double_conv(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        get_activation(activation, inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        get_activation(activation, inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_blocks, in_channels, out_channels, channel_base=64, activation="relu"):
        super().__init__()
        self.n_blocks = n_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(n_blocks):
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2**i,
                                               activation=activation,
                                              ))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(n_blocks-1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
                                                  channel_base * 2**(i-1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(n_blocks-2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2**(i+1),
                                             channel_base * 2**i,
                                             activation=activation,
                                            ))

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.n_blocks-1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i -1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)


class AttentionNet(nn.Module):
    def __init__(self, in_channels, n_blocks, channel_base, activation="relu"):
        super().__init__()
        self.unet = UNet(n_blocks=n_blocks,
                         in_channels=in_channels+1, #->11
                         out_channels=2,
                         channel_base=channel_base,
                         activation=activation,
                        )

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope


# ## 4.2 Monet:

# In[ ]:


class EncoderNet(nn.Module):
    def __init__(self, in_channels, width, height, latent_size=16, mask_mode="concat", activation="relu"):
        super().__init__()
        self.mask_mode = mask_mode
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels + 1 if mask_mode in ["concat", "mulcat"] else in_channels, 32, 3, stride=2),
            get_activation(activation, inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            get_activation(activation, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            get_activation(activation, inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            get_activation(activation, inplace=True),
        )
#64x2x2, maybe small, maybe delete one layer.
        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            get_activation(activation, inplace=True),
            nn.Linear(256, latent_size * 2)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x

class DecoderNet(nn.Module):
    def __init__(self, in_channels, height, width, latent_size, activation="relu"):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(latent_size + 2, 32, 3), # 18 = latent size() + 2
            get_activation(activation, inplace=True),
            nn.Conv2d(32, 32, 3),
            get_activation(activation, inplace=True),
            nn.Conv2d(32, 32, 3),
            get_activation(activation, inplace=True),
            nn.Conv2d(32, 32, 3),
            get_activation(activation, inplace=True),
            nn.Conv2d(32, in_channels + 1, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8) # 8 is for padding
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result


class Monet(nn.Module):
    def __init__(
        self,
        in_channels,
        height,
        width,
        n_blocks,
        channel_base,
        latent_size,
        n_slots,
        bg_sigma,
        fg_sigma,
        zero_color_weight=-1,
        loss_type="gaussian",
        mask_mode="concat",
        activation="relu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_blocks = n_blocks
        self.channel_base = channel_base
        self.latent_size = latent_size
        self.n_slots = n_slots
        self.bg_sigma = bg_sigma
        self.fg_sigma = fg_sigma
        self.loss_type = loss_type
        self.mask_mode = mask_mode
        self.activation = activation
        self.zero_color_weight = zero_color_weight  # weight for the first channel. If -1, will use uniform
        if self.zero_color_weight != -1:
            self.weight = torch.cat([torch.tensor(zero_color_weight)[None], torch.ones(in_channels-1) * ((1-zero_color_weight)/(in_channels-1))])
        self.attention = AttentionNet(
            in_channels = in_channels,
            n_blocks=n_blocks,
            channel_base=channel_base,
            activation=activation,
        )
        self.encoder = EncoderNet(
            in_channels=in_channels,
            width=width,
            height=height,
            latent_size=latent_size,
            mask_mode=mask_mode,
            activation=activation,
        )
        self.decoder = DecoderNet(
            in_channels = in_channels,
            width=width,
            height=height,
            latent_size=latent_size,
            activation=activation,
        )
        self.beta = 0.5
        self.gamma = 0.25

    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.n_slots-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        if self.loss_type == "gaussian":
            p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            sigma = self.bg_sigma if i == 0 else self.fg_sigma
            loss_ele, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            loss += loss_ele + self.beta * kl_z
            if self.loss_type == "gaussian":
                p_xs += loss_ele
            kl_zs += kl_z
            full_reconstruction += mask * x_recon

        masks = torch.cat(masks, 1)
        tr_masks = torch.transpose(masks, 1, 3)
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks

        return {'loss': loss,
                'masks': masks,
                'reconstructions': full_reconstruction}


    def __encoder_step(self, x, mask):
        if self.mask_mode == "concat":
            encoder_input = torch.cat((x, mask), 1)
        elif self.mask_mode == "mul":
            encoder_input = x * mask
        elif self.mask_mode == "mulcat":
            encoder_input = torch.cat((x*mask, mask), 1)
        else:
            raise
        q_params = self.encoder(encoder_input)
        # mlp output is latent size * 2

        means = torch.sigmoid(q_params[:, :self.latent_size]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, self.latent_size:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        mask_pred = decoder_output[:, self.in_channels]
        if self.loss_type == "gaussian":
            x_recon = torch.sigmoid(decoder_output[:, :self.in_channels]) # input channel (10)
            dist = dists.Normal(x_recon, sigma)
            p_x = dist.log_prob(x)
            p_x *= mask
            if self.zero_color_weight == -1:
                p_x = torch.sum(p_x, [1, 2, 3])  #p_x: [B, C, H, W]
            else:
                p_x = torch.sum(p_x, [2, 3])
                p_x = torch.matmul(p_x, self.weight.to(x.device))
            loss_ele = -p_x
        elif self.loss_type == "ce":
            x_recon_logit = decoder_output[:, :self.in_channels]
            if self.zero_color_weight == -1:
                loss_ele = nn.CrossEntropyLoss(reduction='none')(x_recon_logit, x.argmax(1))
            else:
                loss_ele = nn.CrossEntropyLoss(reduction='none', weight=self.weight.to(x.device))(x_recon_logit, x.argmax(1))
            loss_ele *= mask.squeeze(1)  # mask: [B, 1, H, W], loss_ele: [B, H, W]
            loss_ele = torch.sum(loss_ele, [-2,-1])
            x_recon = nn.Softmax(dim=1)(x_recon_logit)
        elif self.loss_type == "gaussian+ce":
            x_recon_logit = decoder_output[:, :self.in_channels]
            x_recon = nn.Softmax(dim=1)(x_recon_logit)

            # gaussian:
            dist = dists.Normal(x_recon, sigma)
            p_x = dist.log_prob(x)
            p_x *= mask
            if self.zero_color_weight == -1:
                p_x = torch.sum(p_x, [1, 2, 3])  #p_x: [B, C, H, W]
            else:
                p_x = torch.sum(p_x, [2, 3])
                p_x = torch.matmul(p_x, self.weight.to(x.device))
            loss_ele_gaussian = -p_x

            # ce:
            if self.zero_color_weight == -1:
                loss_ele_ce = nn.CrossEntropyLoss(reduction='none')(x_recon_logit, x.argmax(1))
            else:
                loss_ele_ce = nn.CrossEntropyLoss(reduction='none', weight=self.weight.to(x.device))(x_recon_logit, x.argmax(1))
            loss_ele_ce *= mask.squeeze(1)
            loss_ele_ce = torch.sum(loss_ele_ce, [-2,-1])

            loss_ele = loss_ele_gaussian + loss_ele_ce
        else:
            raise Exception("loss_type '{}' is not valid!".format(self.loss_type))
        return loss_ele, x_recon, mask_pred

    @property
    def model_dict(self):
        model_dict = {"type": "Monet"}
        model_dict["in_channels"] = self.in_channels
        model_dict["height"] = self.height
        model_dict["width"] = self.width
        model_dict["n_blocks"] = self.n_blocks
        model_dict["channel_base"] = self.channel_base
        model_dict["latent_size"] = self.latent_size
        model_dict["n_slots"] = self.n_slots
        model_dict["bg_sigma"] = self.bg_sigma
        model_dict["fg_sigma"] = self.fg_sigma
        model_dict["mask_mode"] = self.mask_mode
        model_dict["loss_type"] = self.loss_type
        model_dict["activation"] = self.activation
        model_dict["zero_color_weight"] = self.zero_color_weight
        model_dict["state_dict"] = to_cpu(self.state_dict())
        return model_dict


def load_model(model_dict, device="cpu"):
    model_type = model_dict["type"]
    if model_type == "Monet":
        model = Monet(
            in_channels=model_dict["in_channels"],
            height=model_dict["height"],
            width=model_dict["width"],
            n_blocks=model_dict["n_blocks"],
            channel_base=model_dict["channel_base"],
            latent_size=model_dict["latent_size"],
            n_slots=model_dict["n_slots"],
            bg_sigma=model_dict["bg_sigma"],
            fg_sigma=model_dict["fg_sigma"],
            loss_type=model_dict["loss_type"] if "loss_type" in model_dict else "gaussian",
            mask_mode=model_dict["mask_mode"] if "mask_mode" in model_dict else "concat",
            activation=model_dict["activation"] if "activation" in model_dict else "relu",
            zero_color_weight=model_dict["zero_color_weight"] if "zero_color_weight" in model_dict else -1,
        )
    else:
        raise
    model.load_state_dict(model_dict["state_dict"])
    model.to(device)
    return model

