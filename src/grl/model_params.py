l2rpn_case14_sandbox = 904
l2rpn_icaps_2021_large = 2657
l2rpn_idf_2023 = 748


sac_params = {
    "policy": "MultiInputPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,  # 1e6
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "action_noise": None,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "policy_kwargs": {
        "net_arch": [64, 64],
        "optimizer_class": "Adam",
        "activation_fn"
    },
    # "env": "l2rpn_case14_sandbox",
    "verbose": 0,
    "seed": None,
    "device": "auto",
}

gnn_params = {
    "in_channels": 1,
    "hidden_channels": 1,
    "num_layers": 1,
    "out_channels": None,
    "dropout": 0.0,
    "act": "relu",
    "act_first": False,
    "act_kwargs": None,
    "norm": None,
    "norm_kwargs": None,
    "jk": None,
    "aggr": "sum",
    "aggr_kwargs": None,
    "flow": "source_to_target",
    "node_dim": -2,
    "decomposed_layers": 1,
}

gcn_params = {
    "improved": False,
    "cached": False,
    "add_self_loops": None,
    "normalize": True,
    "bias": True,
}

graph_sage = {
    "aggr": "mean",
    "normalize": False,
    "root_weight": True,
    "project": False,
    "bias": True,
}

gat_params = {
    "heads": 1,
    "v2": False,
    "concat": True,
    "negative_slope": 0.2,
    "dropout": 0.0,
    "add_self_loops": True,
    "edge_dim": None,
    "fill_value": "mean",
    "bias": True,
}
