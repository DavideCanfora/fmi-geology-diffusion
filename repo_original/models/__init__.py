from core.praser import init_obj

# Factory functions for models, networks, losses and metrics.
#
# The repository is configuration-driven: run.py does not directly instantiate
# a specific dataset, UNet, loss or model wrapper. Instead, the JSON config
# specifies component names and arguments. These helper functions resolve those
# config entries through init_obj().
#
# For the FMI adaptation, this file is important because new losses, metrics,
# networks or model wrappers can be selected from the config without changing
# the main training script.

def create_model(**cfg_model):
    """
    Instantiate the high-level model/training wrapper.

    The wrapper class is selected by opt["model"]["which_model"].
    In the Palette repository, this is usually models.model.Palette.

    This function injects runtime objects into the model arguments:
        - parsed configuration opt
        - networks
        - dataloaders
        - losses
        - metrics
        - logger/writer

    The returned object owns the training, validation, testing and checkpoint
    logic.
    """
    opt = cfg_model['opt']
    logger = cfg_model['logger']

    model_opt = opt['model']['which_model']
    model_opt['args'].update(cfg_model)
    model = init_obj(model_opt, logger, default_file_name='models.model', init_type='Model')

    return model

def define_network(logger, opt, network_opt):
    """
    Instantiate the diffusion network specified in the config.

    The default network class is models.network.Network, which internally
    builds the selected UNet backbone and diffusion noise schedule.

    During training, weights are initialized according to the init_type field
    in the config. This is the entry point for architectural changes such as
    channel count, UNet width/depth, attention resolution and diffusion schedule.
    """
    net = init_obj(network_opt, logger, default_file_name='models.network', init_type='Network')

    if opt['phase'] == 'train':
        logger.info('Network [{}] weights initialize using [{:s}] method.'.format(net.__class__.__name__, network_opt['args'].get('init_type', 'default')))
        net.init_weights()
    return net


def define_loss(logger, loss_opt):
    """
    Instantiate a training loss from models.loss.

    Losses are selected in the config under opt["model"]["which_losses"].
    Future FMI-specific losses can be added in models/loss.py and selected
    from the JSON config.
    """
    return init_obj(loss_opt, logger, default_file_name='models.loss', init_type='Loss')

def define_metric(logger, metric_opt):
    """
    Instantiate an evaluation metric from models.metric.

    Metrics are selected in the config under opt["model"]["which_metrics"].
    They are used during validation/testing, not to optimize the model unless
    explicitly included in the training loss.
    """
    return init_obj(metric_opt, logger, default_file_name='models.metric', init_type='Metric')
