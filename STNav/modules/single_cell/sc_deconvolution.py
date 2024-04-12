from loguru import logger
from STNav.utils.decorators import pass_STNavCore_params
import inspect
import scvi
import scanpy as sc
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI
import torch


@pass_STNavCore_params
def train_or_load_sc_deconvolution_model(STNavCorePipeline):
    config = STNavCorePipeline.config[STNavCorePipeline.data_type]

    model_types = [
        model_name
        for model_name, model_config in config["model"]["model_type"].items()
        if model_config["usage"]
    ]
    if len(model_types) >= 2:
        raise ValueError(
            logger.error(
                f"Please, choose only 1 model to use. Current active models {model_types = }"
            )
        )
    elif len(model_types) == 0:
        logger.warning(
            f"Returning no model as no models were set to True for training or loading. "
        )
        return None
    model_name = model_types[0]

    adata = sc.read_h5ad(
        STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
            config["model"]["model_type"][model_name]["adata_to_use"]
        ]
    )

    model = eval(model_name)
    # TODO: add assertion that checks if selected layer is normalized or unnormalized counts [0,15,0,23] instead of [0,6.2123,0,8.2123] etc
    model.setup_anndata(
        adata,
        layer=config["model"]["model_type"][model_name]["layer"],
        labels_key=config["DEG"]["rank_genes_groups"]["params"]["groupby"],
    )

    train = config["model"]["model_type"][model_name]["train"]

    if train:
        logger.info(
            f"Training the {model_name} model for deconvolution with '{config['model']['model_type'][model_name]['adata_to_use']}' adata file using the layer {config['model']['model_type'][model_name]['layer']} and the following parameters {config['model']['model_type'][model_name]['params']}."
        )
        sc_model = model(adata)
        logger.info(sc_model.view_anndata_setup())
        training_params = config["model"]["model_type"][model_name]["params"]
        valid_arguments = inspect.signature(sc_model.train).parameters.keys()
        filtered_params = {
            k: v for k, v in training_params.items() if k in valid_arguments
        }
        sc_model.train(**filtered_params)
        sc_model.history["elbo_train"][10:].plot()
        sc_model.save("scmodel", overwrite=True)
    else:
        logger.info(f"Loading the pre-trained {model_name} model for deconvolution.")
        sc_model = model.load(
            config["model"]["pre_trained_model_path"],
            adata,
        )

    # Save to class instance the trained sc_model
    STNavCorePipeline.sc_model = sc_model
