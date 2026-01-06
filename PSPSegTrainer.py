import torch
from torch import distributed as dist
import numpy as np
from torch import nn, autocast
import torch.nn.init as init

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.nets.PSPSeg import PSPSeg, PSPSeg_L, SRM, PRM, RMPruneWrapper
from nnunetv2.nets.STUNet import STUNet_fd

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from time import time, sleep
import gc

def weights_init(m):
    if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight)  # 使用 Kaiming 初始化
        if m.bias is not None:
            init.zeros_(m.bias)
    if isinstance(m, PRM) or isinstance(m, SRM):
        init.zeros_(m.weights)

class Base(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        new_patch_size = (128, 128, 128)
        self.configuration_manager.configuration['patch_size'] = new_patch_size
        self.print_to_log_file("Patch size changed to {}".format(new_patch_size))
        self.plans_manager.plans['configurations'][self.configuration_name]['patch_size'] = new_patch_size

        self.batch_size = 2
        self.configuration_manager.configuration['batch_size'] = self.batch_size
        self.print_to_log_file("Batch size changed to {}".format(self.batch_size))
        self.plans_manager.plans['configurations'][self.configuration_name]['batch_size'] = self.batch_size

        self.mask_value = self.plans_manager.plans['foreground_intensity_properties_per_channel']['0']['percentile_00_5']
        self.print_to_log_file(f"Mask value set to {self.mask_value}")

        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.initial_lr = 5e-4
        self.weight_decay = 3e-5

        self.rl_loss = nn.BCEWithLogitsLoss()

        self.convergence_threshold = 10
        self.prune_step = 1
        self.calib_data_size = 100
        self.dropped_branches = []
        self.drop_fd = False
        self.enable_pruning = False
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    def _get_deep_supervision_scales(self):
        # modify if deep_supervision is enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        self.network.do_ds = enabled
    
    def check_convergence(self):
        if self.current_epoch == 0:
            self.last_prune = 0
        if self.current_epoch <= self.convergence_threshold:
            return False
        losslog = self.logger.my_fantastic_logging['train_losses'][self.last_prune:]
        loss1log = self.logger.my_fantastic_logging['loss1'][self.last_prune:]
        loss2log = self.logger.my_fantastic_logging['loss2'][self.last_prune:]
        if min(losslog) == min(losslog[-self.convergence_threshold:]):
            return False
        if min(loss1log) == min(loss1log[-self.convergence_threshold:]):
            return False
        if min(loss2log) == min(loss2log[-self.convergence_threshold:]):
            return False
        self.print_to_log_file(f"from epoch {self.last_prune} to epoch {self.current_epoch}, min loss:{min(losslog)}, min loss1: {min(loss1log)}, min loss2: {min(loss2log)}")
        self.print_to_log_file(f"best loss: {min(self.logger.my_fantastic_logging['train_losses'])}, best loss1: {min(self.logger.my_fantastic_logging['loss1'])}, best loss2: {min(self.logger.my_fantastic_logging['loss2'])}")
        return True

    def check_over_prune(self):
        losslog = self.logger.my_fantastic_logging['train_losses'][self.last_prune:]
        loss1log = self.logger.my_fantastic_logging['loss1'][self.last_prune:]
        loss2log = self.logger.my_fantastic_logging['loss2'][self.last_prune:]
        if min(self.logger.my_fantastic_logging['train_losses']) + 0.001 <= min(losslog):
            return True
        if min(self.logger.my_fantastic_logging['loss1']) + 0.001 <= min(loss1log):
            return True
        if min(self.logger.my_fantastic_logging['loss2']) + 0.001 <= min(loss2log):
            return True
        return False
    
    def layerwise_prune(self):
        # prune everything that masked
        for module, branches in self.dropped_branches:
            module.prune(branches)
        self.dropped_branches = []
        # find all RMs in the network
        rms = []
        for name, m in self.network.named_modules():
            if isinstance(m, PRM) or isinstance(m, SRM):
                rms.append((name.split('.')[-2], int(name.split('.')[-1])))
        # wrap all RMs with RMPruneWrapper
        prunable_rms = [(seq, idx) for seq, idx in rms if len(getattr(self.network, seq)[idx].blocks) > self.prune_step]
        if prunable_rms == []:
            self.print_to_log_file("No prunable RMs found, pruning stopped!")
            self.enable_pruning = False
            return
        for seq, idx in prunable_rms:
            getattr(self.network, seq)[idx] = RMPruneWrapper(getattr(self.network, seq)[idx], self.prune_step)
        # inference calib_data to cache hidden states
        self.network.eval()
        with torch.no_grad():
            for _ in range(self.calib_data_size):
                data = next(self.dataloader_train)['data']
                data = data.to(self.device, non_blocking=True)
                out = self.network(data)
        # enumerate all RMs and prune them
        for seq, idx in prunable_rms:
            getattr(self.network, seq)[idx].enumerate(self.device)
            branch_to_prune = getattr(self.network, seq)[idx].branch_to_prune
            getattr(self.network, seq)[idx] = getattr(self.network, seq)[idx].prune()
            self.dropped_branches.append((getattr(self.network, seq)[idx], branch_to_prune))
        self.network.train()

    def prune(self):
        # check if need to proceed pruning
        if self.check_convergence():
            if self.check_over_prune():
                self.print_to_log_file("over pruned! start restoring...")
                self.print_to_log_file(f"restored {len(self.dropped_branches) * self.prune_step} blocks")
                for module, branches in self.dropped_branches:
                    module.activate(branches)
                self.dropped_branches = []
                self.print_to_log_file(self.network.get_branches_info())
                self.prune_step -= 1
                if self.prune_step <= 0:
                    self.enable_pruning = False
                    self.print_to_log_file("pruning stopped!")
                else:
                    self.print_to_log_file(f"apply new pruning step {self.prune_step}")
            else:
                self.print_to_log_file(f"pruned {sum([len(branches) for branches in self.dropped_branches])} blocks")
                self.layerwise_prune()
                self.print_to_log_file(f"masked {sum([len(branches) for branches in self.dropped_branches])} blocks")
                torch.cuda.empty_cache()
            self.last_prune = self.current_epoch
            if self.current_epoch >= self.num_epochs*0.4:
                self.enable_pruning = False
                for module, branches in self.dropped_branches:
                    module.activate(branches)
            self.print_to_log_file(self.network.get_branches_info())
            self.print_to_log_file(f"parameters: {sum(p.numel() for p in self.network.parameters())}")

    def on_epoch_start(self):
        if self.enable_pruning:
            self.prune()
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
        torch.cuda.reset_peak_memory_stats(self.device)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            mask = [(gt != 0).float() for gt in target]
            masked_data = torch.where(mask[0].bool(), data, self.mask_value)

            dec_out, output, l1 = self.network(data, masked_data)
            
            # del data
            l2 = self.rl_loss(dec_out, mask[1])
            if self.drop_fd:
                l = self.loss(output, target)
            else:
                l = self.loss(output, target) + (l1 + l2) * 0.1
        
        # debug
        # for name, param in self.network.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"Gradient for {name} contains NaN!")
        #         if torch.isinf(param.grad).any():
        #             print(f"Gradient for {name} contains Inf!")
        # if torch.isnan(l).any():
        #     self.print_to_log_file(f"loss={l.item()}, loss1={l1.item()}, loss2={l2.item()}")


        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy(), 'loss1': l1.detach().cpu().numpy(), 'loss2': l2.detach().cpu().numpy()}
    
    def on_train_epoch_end(self, train_outputs):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()

            loss1_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(loss1_tr, outputs['loss1'])
            loss1_here = np.vstack(loss1_tr).mean()

            loss2_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(loss2_tr, outputs['loss2'])
            loss2_here = np.vstack(loss2_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            loss1_here = np.mean(outputs['loss1'])
            loss2_here = np.mean(outputs['loss2'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('loss1', loss1_here, self.current_epoch)
        self.logger.log('loss2', loss2_here, self.current_epoch)
        self.print_to_log_file(f"Max allocated memory: {torch.cuda.max_memory_allocated(self.device) / (1024 ** 2):.2f} MB")
        # self.network.print_GRM_weights()
    

    def on_train_end(self):
        super().on_train_end()
        torch.save(self.network, join(self.output_folder, "overall_arch.pth"))
    
class prune_para(Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.enable_pruning = True
        self.initial_lr = 1e-3
        self.prune_step = 2

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        self.network = torch.load(join(self.output_folder, "overall_arch.pth"), weights_only=False)
        
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # self.model_type = "PSPSeg_S"
        # model = PSPSeg(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256],
        #               block_counts=[1, 1, 1, 1, 1],
        #               exp_r=0.5,
        #               para_num=4,
        #               deep_supervision=enable_deep_supervision)
        # model.load_pretrained_ckpt("/media/userdisk0/lhli/nnUNet-master/pretrained_model/PSPSeg_s_ep1k.model")

        # self.model_type = "PSPSeg_B"
        model = PSPSeg(in_channels=num_input_channels,
                      out_channels=num_output_channels,
                      redundant_module=PRM,
                      n_channels=[16, 32, 64, 128, 256],
                      block_counts=[1, 1, 1, 1, 1],
                      exp_r=2,
                      para_num=7,
                      deep_supervision=enable_deep_supervision)
        # model.load_pretrained_ckpt("/media/userdisk0/lhli/nnUNet-master/pretrained_model/PSPSeg_b_ep1k.model")

        # self.model_type = "PSPSeg_L"
        # model = PSPSeg_L(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256, 320],
        #               block_counts=[1, 1, 1, 1, 1, 1],
        #               exp_r=2,
        #               para_num=7,
        #               deep_supervision=enable_deep_supervision)
        # model.load_pretrained_ckpt("/media/userdisk0/lhli/nnUNet-master/pretrained_model/PSPSeg_l_ep1k.model")

        # self.model_type = "PSPSeg_H"
        # model = PSPSeg_L(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[32, 64, 128, 256, 320, 512],
        #               block_counts=[1, 1, 1, 1, 1, 1],
        #               exp_r=2,
        #               para_num=7,
        #               deep_supervision=enable_deep_supervision)
        # model.load_pretrained_ckpt("/media/userdisk0/lhli/nnUNet-master/pretrained_model/PSPSeg_l_ep1k.model")
        
        return model
    
class ablation1(Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.enable_pruning = False
        self.drop_fd = True
        self.initial_lr = 1e-3
        
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # self.model_type = "PSPSeg_S"
        # model = PSPSeg(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256],
        #               block_counts=[1, 1, 1, 1, 1],
        #               exp_r=0.5,
        #               para_num=1,
        #               deep_supervision=enable_deep_supervision)

        # self.model_type = "PSPSeg_B"
        # model = PSPSeg(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256],
        #               block_counts=[1, 1, 1, 1, 1],
        #               exp_r=2,
        #               para_num=1,
        #               deep_supervision=enable_deep_supervision)

        # self.model_type = "PSPSeg_L"
        model = PSPSeg_L(in_channels=num_input_channels,
                      out_channels=num_output_channels,
                      redundant_module=PRM,
                      n_channels=[16, 32, 64, 128, 256, 320],
                      block_counts=[1, 1, 1, 1, 1, 1],
                      exp_r=2,
                      para_num=1,
                      deep_supervision=enable_deep_supervision)
        
        return model
    
class ablation2(Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.enable_pruning = False
        self.drop_fd = True
        self.initial_lr = 1e-3
        
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # self.model_type = "PSPSeg_S"
        # model = PSPSeg(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256],
        #               block_counts=[1, 1, 1, 1, 1],
        #               exp_r=0.5,
        #               para_num=4,
        #               deep_supervision=enable_deep_supervision)

        # self.model_type = "PSPSeg_B"
        model = PSPSeg(in_channels=num_input_channels,
                      out_channels=num_output_channels,
                      redundant_module=PRM,
                      n_channels=[16, 32, 64, 128, 256],
                      block_counts=[1, 1, 1, 1, 1],
                      exp_r=2,
                      para_num=7,
                      deep_supervision=enable_deep_supervision)

        # self.model_type = "PSPSeg_L"
        # model = PSPSeg_L(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256, 320],
        #               block_counts=[1, 1, 1, 1, 1, 1],
        #               exp_r=2,
        #               para_num=7,
        #               deep_supervision=enable_deep_supervision)
        
        return model
    
class ablation3(Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.enable_pruning = False
        self.drop_fd = False
        self.initial_lr = 1e-3
        
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # self.model_type = "PSPSeg_S"
        # model = PSPSeg(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256],
        #               block_counts=[1, 1, 1, 1, 1],
        #               exp_r=0.5,
        #               para_num=4,
        #               deep_supervision=enable_deep_supervision)

        # self.model_type = "PSPSeg_B"
        # model = PSPSeg(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256],
        #               block_counts=[1, 1, 1, 1, 1],
        #               exp_r=2,
        #               para_num=7,
        #               deep_supervision=enable_deep_supervision)

        # self.model_type = "PSPSeg_L"
        model = PSPSeg_L(in_channels=num_input_channels,
                      out_channels=num_output_channels,
                      redundant_module=PRM,
                      n_channels=[16, 32, 64, 128, 256, 320],
                      block_counts=[1, 1, 1, 1, 1, 1],
                      exp_r=2,
                      para_num=7,
                      deep_supervision=enable_deep_supervision)
        
        return model
    
class ablation4(Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.enable_pruning = True
        self.drop_fd = False
        self.initial_lr = 1e-3
        self.prune_step = 1
        
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # self.model_type = "PSPSeg_S"
        model = PSPSeg(in_channels=num_input_channels,
                      out_channels=num_output_channels,
                      redundant_module=PRM,
                      n_channels=[16, 32, 64, 128, 256],
                      block_counts=[1, 1, 1, 1, 1],
                      exp_r=0.5,
                      para_num=4,
                      deep_supervision=enable_deep_supervision)

        # self.model_type = "PSPSeg_B"
        # model = PSPSeg(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256],
        #               block_counts=[1, 1, 1, 1, 1],
        #               exp_r=2,
        #               para_num=7,
        #               deep_supervision=enable_deep_supervision)

        # self.model_type = "PSPSeg_L"
        # model = PSPSeg_L(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=PRM,
        #               n_channels=[16, 32, 64, 128, 256, 320],
        #               block_counts=[1, 1, 1, 1, 1, 1],
        #               exp_r=2,
        #               para_num=7,
        #               deep_supervision=enable_deep_supervision)
        
        return model
    
class retrain_s(Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_pruning = False
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # type0: pretrained model pruning, retrain
        # model = torch.load("/media/userdisk0/lhli/nnUNet-master/nnUNet_results/Dataset800_Liver/prune_para/S_pretrain_prune/overall_arch.pth", weights_only=False)
        # model.load_pretrained_ckpt("/media/userdisk0/lhli/nnUNet-master/pretrained_model/PSPSeg_s_ep1k.model")

        # type1: directly model pruning, retrain
        model = torch.load("/media/userdisk0/lhli/nnUNet-master/nnUNet_results/Dataset800_Liver/ablation4/exp_0/overall_arch.pth", weights_only=False)
        model.apply(weights_init)
        
        return model
    
class retrain_b(Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_pruning = False
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # type0: pretrained model pruning, retrain
        # model = torch.load("/media/userdisk0/lhli/nnUNet-master/nnUNet_results/Dataset800_Liver/prune_para/B_pretrain_prune/overall_arch.pth", weights_only=False)
        # model.load_pretrained_ckpt("/media/userdisk0/lhli/nnUNet-master/pretrained_model/PSPSeg_b_ep1k.model")

        # type1: directly model pruning, retrain
        model = torch.load("/media/userdisk0/lhli/nnUNet-master/nnUNet_results/Dataset800_Liver/ablation4/exp_1/overall_arch.pth", weights_only=False)
        model.apply(weights_init)
        
        return model
    
class retrain_l(Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_pruning = False
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # type0: pretrained model pruning, retrain
        # model = torch.load("/media/userdisk0/lhli/nnUNet-master/nnUNet_results/Dataset800_Liver/prune_para/L_pretrain_prune/overall_arch.pth", weights_only=False)
        # model.load_pretrained_ckpt("/media/userdisk0/lhli/nnUNet-master/pretrained_model/PSPSeg_l_ep1k_backup.model")

        # type1: directly model pruning, retrain
        model = torch.load("/media/userdisk0/lhli/nnUNet-master/nnUNet_results/Dataset800_Liver/ablation4/exp_2/overall_arch.pth", weights_only=False)
        model.apply(weights_init)
        
        return model
    
class prune_seq(Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_epochs = 1000
        self.initial_lr = 1e-3
        self.enable_pruning = True
        self.prune_step = 2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        self.network = torch.load(join(self.output_folder, "overall_arch.pth"), weights_only=False)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # self.model_type = "PSPSeg_S"
        model = PSPSeg(in_channels=num_input_channels,
                      out_channels=num_output_channels,
                      redundant_module=SRM,
                      n_channels=[16, 32, 64, 128, 256],
                      block_counts=[1, 1, 1, 1, 1],
                      exp_r=0.5,
                      para_num=4,
                      deep_supervision=enable_deep_supervision)
        
        # self.model_type = "PSPSeg_B"
        # model = PSPSeg(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=SRM,
        #               n_channels=[16, 32, 64, 128, 256],
        #               block_counts=[1, 1, 1, 1, 1],
        #               exp_r=2,
        #               para_num=7,
        #               deep_supervision=enable_deep_supervision)
        
        # self.model_type = "PSPSeg_L"
        # model = PSPSeg_L(in_channels=num_input_channels,
        #               out_channels=num_output_channels,
        #               redundant_module=SRM,
        #               n_channels=[16, 32, 64, 128, 256, 320],
        #               block_counts=[1, 1, 1, 1, 1, 1],
        #               exp_r=2,
        #               para_num=7,
        #               deep_supervision=enable_deep_supervision)
        
        return model
