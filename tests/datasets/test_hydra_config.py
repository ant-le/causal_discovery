from omegaconf import OmegaConf

from causal_meta.datasets.data_module import CausalMetaModule


def test_causal_meta_module_accepts_target_based_configs() -> None:


    cfg = OmegaConf.create(


        {


            "train_family": {


                "name": "train",


                "n_nodes": 4,


                "graph_cfg": {


                    "_target_": "causal_meta.datasets.generators.graphs.er.ErdosRenyiGenerator",


                    "sparsity": 0.3,


                },


                "mech_cfg": {


                    "_target_": "causal_meta.datasets.generators.mechanisms.linear.LinearMechanismFactory",


                    "weight_scale": 0.1,


                },


            },


            "test_families": {


                "test": {


                    "name": "test",


                    "n_nodes": 4,


                    "graph_cfg": {


                        "_target_": "causal_meta.datasets.generators.graphs.sf.ScaleFreeGenerator",


                        "m": 2,


                    },


                    "mech_cfg": {


                        "_target_": "causal_meta.datasets.generators.mechanisms.mlp.MLPMechanismFactory",


                        "hidden_dim": 8,


                    },


                }


            },


            "seeds_train": [0, 1, 2],


            "seeds_test": [10, 11, 12],


            "samples_per_task": 4,


        }


    )





    module = CausalMetaModule.from_config(cfg)


    module.setup()





    assert module.train_family is not None


    assert "test" in module.test_families


    assert module.train_dataset is not None


    assert "test" in module.test_datasets








def test_causal_meta_module_from_config_parses_top_level_cfg() -> None:


    cfg = OmegaConf.create(


        {


            "train_family": {


                "name": "train",


                "n_nodes": 4,


                "graph_cfg": {


                    "_target_": "causal_meta.datasets.generators.graphs.er.ErdosRenyiGenerator",


                    "sparsity": 0.3,


                },


                "mech_cfg": {


                    "_target_": "causal_meta.datasets.generators.mechanisms.linear.LinearMechanismFactory",


                    "weight_scale": 0.1,


                },


            },


            "test_families": {


                "test": {


                    "name": "test",


                    "n_nodes": 4,


                    "graph_cfg": {


                        "_target_": "causal_meta.datasets.generators.graphs.sf.ScaleFreeGenerator",


                        "m": 2,


                    },


                    "mech_cfg": {


                        "_target_": "causal_meta.datasets.generators.mechanisms.mlp.MLPMechanismFactory",


                        "hidden_dim": 8,


                    },


                }


            },


            "seeds_train": [0, 1, 2],


            "seeds_test": [10, 11, 12],


            "samples_per_task": 4,


        }


    )








    module = CausalMetaModule.from_config(cfg)


    module.setup()





    assert module.train_dataset is not None


    assert "test" in module.test_datasets

