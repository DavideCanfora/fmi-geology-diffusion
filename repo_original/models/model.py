import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy

# High-level Palette training wrapper.
#
# This file does not define the UNet architecture or the diffusion equations.
# Instead, it wraps the network with:
# - optimizer creation;
# - checkpoint loading/saving;
# - exponential moving average;
# - train/validation/test loops;
# - logging and image export.
#
# The actual diffusion forward loss is computed by models/network.py.

class EMA():
    """
    Exponential Moving Average of model parameters.

    During training, an EMA copy of the generator can be updated from the
    current network weights. EMA weights are often more stable for sampling
    in diffusion models.
    """
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        """
        Initialize the Palette training wrapper.

        This class receives already-created networks, losses, dataloaders,
        metrics, logger and writer from the factory functions. It configures
        the optimizer, optional EMA model, checkpoint loading, diffusion loss
        and noise schedule.

        For FMI inpainting, the training batch contains gt_image, cond_image
        and mask produced by FMIInpaintDataset.
        """
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.sample_num = sample_num
        self.task = task
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training() 

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        metric_names = [m.__name__ for m in self.metrics]
        if self.task in ['inpainting', 'uncropping']:
            metric_names = metric_names + ['masked_mae']
        self.val_metrics = LogTracker(*metric_names, phase='val')
        self.test_metrics = LogTracker(*metric_names, phase='test')

        
    def set_input(self, data):
        """
        Move a batch from the dataloader into model attributes.

        Expected fields for inpainting:
            cond_image : conditional masked/noisy image
            gt_image   : target image
            mask       : binary inpainting mask
            mask_image : visualization image
            path       : filename

        FMIInpaintDataset also returns valid_region, but the current Palette
        wrapper does not use it yet. It can be introduced later for
        FMI-specific masked/valid-region losses.
        """
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }
        if self.task in ['inpainting','uncropping']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image+1)/2,
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def compute_masked_mae(self):
        """
        Compute MAE only inside the inpainting mask.

        This is more informative than full-image MAE for FMI inpainting,
        because the target region is the deliberately hidden valid area.
        The mask has shape [B, 1, H, W] and is broadcast over RGB channels.
        """
        eps = 1e-8
        diff = (self.gt_image - self.output).abs()
        masked_diff = diff * self.mask
        denom = self.mask.sum() * diff.shape[1] + eps
        return masked_diff.sum() / denom

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        
        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        """
        Execute one training epoch over phase_loader.

        For each batch:
        1. load gt_image, cond_image and mask;
        2. compute the diffusion training loss through self.netG(...);
        3. backpropagate;
        4. update network weights;
        5. log scalar losses and visual examples;
        6. update EMA weights if enabled.

        The scalar loss is produced inside models/network.py.
        """
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        """
        Run validation sampling/restoration.

        Unlike train_step, validation does not optimize the network. It calls
        self.netG.restoration(...) to generate completed images from the
        conditional input and mask, computes metrics against gt_image, and
        saves visual results.
        """
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = (
                            self.netG.module.restoration_repaint(
                                self.cond_image, y_t=self.cond_image, y_0=self.gt_image,
                                mask=self.mask, sample_num=self.sample_num,
                                jump_length=self.repaint_jump_length,
                                jump_n_sample=self.repaint_jump_n_sample
                            )
                            if self.sampling_mode == 'repaint'
                            else self.netG.module.restoration(
                                self.cond_image, y_t=self.cond_image,
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num
                            )
                        )
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = (
                            self.netG.restoration_repaint(
                                self.cond_image, y_t=self.cond_image, y_0=self.gt_image,
                                mask=self.mask, sample_num=self.sample_num,
                                jump_length=self.repaint_jump_length,
                                jump_n_sample=self.repaint_jump_n_sample
                            )
                            if self.sampling_mode == 'repaint'
                            else self.netG.restoration(
                                self.cond_image, y_t=self.cond_image,
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num
                            )
                        )
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                    
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)

                if self.task in ['inpainting', 'uncropping'] and self.mask is not None:
                    masked_mae = self.compute_masked_mae()
                    self.val_metrics.update('masked_mae', masked_mae.item())
                    self.writer.add_scalar('masked_mae', masked_mae.item())

                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        """
        Run test-time restoration over the selected phase dataset.

        This follows the same restoration path as validation, but iterates over
        phase_loader and writes final test metrics/results.
        """
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = (
                            self.netG.module.restoration_repaint(
                                self.cond_image, y_t=self.cond_image, y_0=self.gt_image,
                                mask=self.mask, sample_num=self.sample_num,
                                jump_length=self.repaint_jump_length,
                                jump_n_sample=self.repaint_jump_n_sample
                            )
                            if self.sampling_mode == 'repaint'
                            else self.netG.module.restoration(
                                self.cond_image, y_t=self.cond_image,
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num
                            )
                        )
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = (
                            self.netG.restoration_repaint(
                                self.cond_image, y_t=self.cond_image, y_0=self.gt_image,
                                mask=self.mask, sample_num=self.sample_num,
                                jump_length=self.repaint_jump_length,
                                jump_n_sample=self.repaint_jump_n_sample
                            )
                            if self.sampling_mode == 'repaint'
                            else self.netG.restoration(
                                self.cond_image, y_t=self.cond_image,
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num
                            )
                        )
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)

                if self.task in ['inpainting', 'uncropping'] and self.mask is not None:
                    masked_mae = self.compute_masked_mae()
                    self.test_metrics.update('masked_mae', masked_mae.item())
                    self.writer.add_scalar('masked_mae', masked_mae.item())

                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()
