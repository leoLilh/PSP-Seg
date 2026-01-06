import torch.nn as nn
import torch
import itertools as I

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.convs(x)

class LNB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 exp_r=0.5):
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = tuple([kernel_size for _ in range(3)])
        padding = tuple([k//2 for k in kernel_size])
        hidden_channels = int(in_channels*exp_r)
        
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, padding=0),
            nn.InstanceNorm3d(hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.convs(x)
    
class DownSample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_r=0.5):
        super().__init__()
        
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
        )
        self.bypassconv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.bypassconv(x) + self.convs(x)
        return self.act(x)
    
class UpSample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_r=0.5):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x
    
class RM(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=0.5, para_num=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert para_num in [7,4,1], "Unsupported para_num"
        if para_num == 7:
            self.blocks = nn.ModuleList([
                LNB(in_channels, out_channels, kernel_size=(1,1,1), exp_r=exp_r),
                LNB(in_channels, out_channels, kernel_size=(1,1,3), exp_r=exp_r),
                LNB(in_channels, out_channels, kernel_size=(1,3,1), exp_r=exp_r),
                LNB(in_channels, out_channels, kernel_size=(3,1,1), exp_r=exp_r),
                LNB(in_channels, out_channels, kernel_size=(1,3,3), exp_r=exp_r),
                LNB(in_channels, out_channels, kernel_size=(3,1,3), exp_r=exp_r),
                LNB(in_channels, out_channels, kernel_size=(3,3,1), exp_r=exp_r),
            ])
        elif para_num == 4:
            self.blocks = nn.ModuleList([
                LNB(in_channels, out_channels, kernel_size=(1,1,1), exp_r=exp_r),
                LNB(in_channels, out_channels, kernel_size=(1,3,3), exp_r=exp_r),
                LNB(in_channels, out_channels, kernel_size=(3,1,3), exp_r=exp_r),
                LNB(in_channels, out_channels, kernel_size=(3,3,1), exp_r=exp_r),
            ])
        elif para_num == 1:
            self.blocks = nn.ModuleList([
                LNB(in_channels, out_channels, kernel_size=(1,1,1), exp_r=exp_r),
            ])
        self.weights = nn.Parameter(torch.zeros(len(self.blocks)))
        self.register_buffer('active_paths', torch.ones(len(self.blocks), dtype=torch.bool))

    def mask(self, idx:int|list|tuple):
        if isinstance(idx, list) or isinstance(idx, tuple):
            for i in idx:
                self.active_paths[i] = False
        else:
            self.active_paths[idx] = False
            
    def activate(self, idx:int|list|tuple):
        if isinstance(idx, list) or isinstance(idx, tuple):
            for i in idx:
                self.active_paths[i] = True
        else:
            self.active_paths[idx] = True

    def prune(self, idx:int|list|tuple):
        if isinstance(idx, list) or isinstance(idx, tuple):
            for i in idx:
                self.active_paths[i] = False
                self.blocks[i] = None
        else:
            self.active_paths[idx] = False
            self.blocks[idx] = None
    
class SRM(RM):
    def __init__(self, in_channels, out_channels, exp_r=0.5, para_num=7):
        super().__init__(in_channels, out_channels, exp_r, para_num)
    
    def forward(self, x):
        for i in range(len(self.blocks)):
            if self.active_paths[i]:
                x = self.blocks[i](x) * self.weights[i] + x
        return x
    
class PRM(RM):
    def __init__(self, in_channels, out_channels, exp_r=0.5, para_num=7):
        super().__init__(in_channels, out_channels, exp_r, para_num)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        if self.active_paths.sum().item() <= 0:
            return x
        expert_outputs = torch.stack([block(x)*weight for active, block, weight in zip(self.active_paths, self.blocks, self.weights) if active], dim=1) # [B, k, C, D, H, W]
        output = torch.sum(expert_outputs, dim=1)
        return self.act(output + x)
    
class RMPruneWrapper(nn.Module):
    def __init__(self, module:RM, prune_num):
        super().__init__()
        self.module = module
        self.prune_num = prune_num
        self.branch_to_prune = None
        self.cache_data = []

    def forward(self, x):
        y = self.module(x)
        self.cache_data.append((x.to('cpu'), y.to('cpu')))
        return y
    
    @torch.no_grad()
    def enumerate(self, device):
        loss_history = {}
        for dropped in I.combinations([i for i in range(len(self.module.blocks)) if self.module.active_paths[i]], self.prune_num):
            self.module.mask(dropped)
            loss = 0
            for cache_in, cache_out in self.cache_data:
                out = self.module(cache_in.to(device))
                loss += torch.norm(cache_out.to(device) - out).item()
            loss_history[dropped] = loss
            self.module.activate(dropped)
        if len(loss_history) > 0:
            self.branch_to_prune = min(loss_history, key=loss_history.get)
        return loss_history
    
    def prune(self):
        if self.branch_to_prune is not None:
            self.module.mask(self.branch_to_prune)
        return self.module
    
class PSPSeg_nofd(nn.Module):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        redundant_module = PRM,
        n_channels: list = [16, 32, 64, 128, 256],
        block_counts: list = [1,1,1,1,1],
        dim = '3d',                                # 2d or 3d
        exp_r: float = 0.5,                            # Expansion ratio as in Swin Transformers
        para_num: int = 4,
        deep_supervision: bool = False,             # Can be used to test deep supervision
    ):
        super().__init__()

        self.do_ds = deep_supervision
        assert dim in ['2d', '3d']
        assert len(n_channels) == len(block_counts)

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
            
        self.stem = conv(in_channels, n_channels[0], kernel_size=1)
        
        self.enc_block_0 = nn.Sequential(*[
            ConvBlock(
                in_channels=n_channels[0],
                out_channels=n_channels[0]
            )
            for i in range(block_counts[0])]
        ) 

        self.down_0 = DownSample(
            in_channels=n_channels[0],
            out_channels=n_channels[1],
            exp_r=exp_r
        )
    
        self.enc_block_1 = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[1],
                out_channels=n_channels[1],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[1])]
        ) 

        self.down_1 = DownSample(
            in_channels=n_channels[1],
            out_channels=n_channels[2],
            exp_r=exp_r
        )

        self.enc_block_2 = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[2],
                out_channels=n_channels[2],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[2])]
        ) 

        self.down_2 = DownSample(
            in_channels=n_channels[2],
            out_channels=n_channels[3],
            exp_r=exp_r
        )
        
        self.enc_block_3 = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[3],
                out_channels=n_channels[3],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[3])]
        ) 
        
        self.down_3 = DownSample(
            in_channels=n_channels[3],
            out_channels=n_channels[4],
            exp_r=exp_r
        )

        self.bottleneck = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[4],
                out_channels=n_channels[4],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[4])]
        ) 

        self.up_3 = UpSample(
            in_channels=n_channels[4],
            out_channels=n_channels[3],
            exp_r=exp_r
        )

        self.dec_block_3 = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[3],
                out_channels=n_channels[3],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[3])]
        ) 

        self.up_2 = UpSample(
            in_channels=n_channels[3],
            out_channels=n_channels[2],
            exp_r=exp_r
        )

        self.dec_block_2 = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[2],
                out_channels=n_channels[2],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[2])]
        ) 

        self.up_1 = UpSample(
            in_channels=n_channels[2],
            out_channels=n_channels[1],
            exp_r=exp_r
        )

        self.dec_block_1 = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[1],
                out_channels=n_channels[1],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[1])]
        ) 

        self.up_0 = UpSample(
            in_channels=n_channels[1],
            out_channels=n_channels[0],
            exp_r=exp_r
        )

        self.dec_block_0 = nn.Sequential(*[
            ConvBlock(
                in_channels=n_channels[0],
                out_channels=n_channels[0]
            )
            for i in range(block_counts[0])]
        ) 

        self.out_0 = conv(in_channels=n_channels[0], out_channels=out_channels, kernel_size=1)

        if deep_supervision:
            self.out_1 = conv(in_channels=n_channels[1], out_channels=out_channels, kernel_size=1)
            self.out_2 = conv(in_channels=n_channels[2], out_channels=out_channels, kernel_size=1)
            self.out_3 = conv(in_channels=n_channels[3], out_channels=out_channels, kernel_size=1)
            self.out_4 = conv(in_channels=n_channels[4], out_channels=out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3 
        x = self.dec_block_3(dec_x)
        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2 
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1 
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0 
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3]
        else: 
            return x
        
    def load_pretrained_ckpt(self, ckpt_path:str):
        print(f"loading from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        pretrain_dict = ckpt['network_weights'] if 'network_weights' in ckpt else ckpt['state_dict']
        model_dict = self.state_dict()
        
        keys_to_remove = [key for key in pretrain_dict.keys() if key.find('stem') != -1 or key.find('out') != -1]
        # do not load active_paths when retraining
        # keys_to_remove += [key for key in pretrain_dict.keys() if key.find('active_paths') != -1]
        for key in keys_to_remove:
            del pretrain_dict[key]
        pretrain_dict['up_4.conv.weight'] = pretrain_dict['up_4.conv.weight'][:,:-1,:,:,:]

        pretrain_dict_load = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        pretrain_dict_not_load = [k for k in pretrain_dict.keys() if k not in model_dict]
        print("no update: ", pretrain_dict_not_load + keys_to_remove)
        print("[pretrain_%d/model_%d]: %d loaded layers" % (len(pretrain_dict), len(model_dict), len(pretrain_dict_load)))
        model_dict.update(pretrain_dict_load)
        self.load_state_dict(model_dict)

        print(sum(p.numel() for p in self.parameters()))
    
    def get_branches_info(self):
        branches = []
        for name, module in self.named_modules():
            if isinstance(module, PRM) or isinstance(module, SRM):
                branches.append(f"{name}: {module.active_paths.tolist()}")
        return branches

class PSPSeg(PSPSeg_nofd):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        redundant_module = PRM, 
        n_channels: list = [16, 32, 64, 128, 256],
        block_counts: list = [1,1,1,1,1],
        dim = '3d',                                # 2d or 3d
        exp_r: float = 0.5,                            # Expansion ratio as in Swin Transformers
        para_num: int = 4,
        deep_supervision: bool = False,             # Can be used to test deep supervision
    ):
        super().__init__(in_channels,
                         out_channels,
                         redundant_module,
                         n_channels,
                         block_counts,
                         dim,
                         exp_r,
                         para_num,
                         deep_supervision)

    def forward(self, x, x_masked=None):
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)

        if self.training:
            with torch.no_grad():
                x_masked = self.stem(x_masked)
                x_masked_res_0 = self.enc_block_0(x_masked)
                x_masked = self.down_0(x_masked_res_0)
                x_masked_res_1 = self.enc_block_1(x_masked)
                x_masked = self.down_1(x_masked_res_1)
                x_masked_res_2 = self.enc_block_2(x_masked)
                x_masked = self.down_2(x_masked_res_2)
                x_masked_res_3 = self.enc_block_3(x_masked)
                x_masked = self.down_3(x_masked_res_3)
                x_masked = self.bottleneck(x_masked).detach()

            loss_tp = (1 - torch.nn.functional.cosine_similarity(x, x_masked, dim=1)).mean() + \
                      (1 - torch.nn.functional.cosine_similarity(x_res_3, x_masked_res_3, dim=1)).mean() + \
                      (1 - torch.nn.functional.cosine_similarity(x_res_2, x_masked_res_2, dim=1)).mean() + \
                      (1 - torch.nn.functional.cosine_similarity(x_res_1, x_masked_res_1, dim=1)).mean() + \
                      (1 - torch.nn.functional.cosine_similarity(x_res_0, x_masked_res_0, dim=1)).mean()

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3 
        x = self.dec_block_3(dec_x)
        if self.do_ds:
            # dec_out_3 = (torch.max(torch.abs(x), dim=1)[0]).unsqueeze(1)
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2 
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            # dec_out_2 = (torch.max(torch.abs(x), dim=1)[0]).unsqueeze(1)
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1 
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            # dec_out_1 = (torch.max(torch.abs(x), dim=1)[0]).unsqueeze(1)
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1
        
        # dec_out = (torch.max(torch.abs(x), dim=1)[0]).unsqueeze(1)
        # dec_out = (torch.max(x, dim=1)[0]).unsqueeze(1)
        dec_out = x.mean(dim=1, keepdim=True)

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0 
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.training:
            if self.do_ds:
                # return [dec_out, dec_out_1, dec_out_2, dec_out_3], [x, x_ds_1, x_ds_2, x_ds_3], loss_tp
                return dec_out, [x, x_ds_1, x_ds_2, x_ds_3], loss_tp
            else: 
                return dec_out, x, loss_tp
        else:
            if self.do_ds:
                return [x, x_ds_1, x_ds_2, x_ds_3]
            else:
                return x
        
class PSPSeg_L_nofd(PSPSeg_nofd):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        redundant_module = PRM, 
        n_channels: list = [16, 32, 64, 128, 256, 320],
        block_counts: list = [1,1,1,1,1,1],
        dim = '3d',                                # 2d or 3d
        exp_r: float = 2,                            # Expansion ratio as in Swin Transformers
        para_num: int = 7,
        deep_supervision: bool = False,             # Can be used to test deep supervision
    ):
        super().__init__(in_channels,
                         out_channels,
                         redundant_module,
                         n_channels,
                         block_counts,
                         dim,
                         exp_r,
                         para_num,
                         deep_supervision)

        self.enc_block_4 = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[4],
                out_channels=n_channels[4],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[4])]
        ) 
        
        self.down_4 = DownSample(
            in_channels=n_channels[4],
            out_channels=n_channels[5],
            exp_r=exp_r
        )

        self.bottleneck = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[5],
                out_channels=n_channels[5],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[5])]
        ) 

        self.up_4 = UpSample(
            in_channels=n_channels[5],
            out_channels=n_channels[4],
            exp_r=exp_r
        )

        self.dec_block_4 = nn.Sequential(*[
            redundant_module(
                in_channels=n_channels[4],
                out_channels=n_channels[4],
                exp_r=exp_r,
                para_num=para_num,
            )
            for i in range(block_counts[4])]
        ) 

    def forward(self, x, x_masked=None):
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)
        x_res_4 = self.enc_block_4(x)
        x = self.down_4(x_res_4)

        x = self.bottleneck(x)

        x_up_4 = self.up_4(x)
        dec_x = x_res_4 + x_up_4 
        x = self.dec_block_4(dec_x)
        if self.do_ds:
            x_ds_4 = self.out_4(x)
        del x_res_4, x_up_4

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3 
        x = self.dec_block_3(dec_x)
        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2 
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1 
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0 
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else: 
            return x

class PSPSeg_L(PSPSeg_L_nofd):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        redundant_module = PRM, 
        n_channels: list = [16, 32, 64, 128, 256, 320],
        block_counts: list = [1,1,1,1,1,1],
        dim = '3d',                                # 2d or 3d
        exp_r: float = 2,                            # Expansion ratio as in Swin Transformers
        para_num: int = 7,
        deep_supervision: bool = False,             # Can be used to test deep supervision
    ):
        super().__init__(in_channels,
                         out_channels,
                         redundant_module,
                         n_channels,
                         block_counts,
                         dim,
                         exp_r,
                         para_num,
                         deep_supervision)

    def forward(self, x, x_masked=None):
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)
        x_res_4 = self.enc_block_4(x)
        x = self.down_4(x_res_4)

        x = self.bottleneck(x)

        if self.training:
            with torch.no_grad():
                x_masked = self.stem(x_masked)
                x_masked_res_0 = self.enc_block_0(x_masked)
                x_masked = self.down_0(x_masked_res_0)
                x_masked_res_1 = self.enc_block_1(x_masked)
                x_masked = self.down_1(x_masked_res_1)
                x_masked_res_2 = self.enc_block_2(x_masked)
                x_masked = self.down_2(x_masked_res_2)
                x_masked_res_3 = self.enc_block_3(x_masked)
                x_masked = self.down_3(x_masked_res_3)
                x_masked_res_4 = self.enc_block_4(x_masked)
                x_masked = self.down_4(x_masked_res_4)
                x_masked = self.bottleneck(x_masked).detach()

            loss_tp = (1 - torch.nn.functional.cosine_similarity(x, x_masked, dim=1)).mean() + \
                      (1 - torch.nn.functional.cosine_similarity(x_res_4, x_masked_res_4, dim=1)).mean() + \
                      (1 - torch.nn.functional.cosine_similarity(x_res_3, x_masked_res_3, dim=1)).mean() + \
                      (1 - torch.nn.functional.cosine_similarity(x_res_2, x_masked_res_2, dim=1)).mean() + \
                      (1 - torch.nn.functional.cosine_similarity(x_res_1, x_masked_res_1, dim=1)).mean() + \
                      (1 - torch.nn.functional.cosine_similarity(x_res_0, x_masked_res_0, dim=1)).mean()

        x_up_4 = self.up_4(x)
        dec_x = x_res_4 + x_up_4 
        x = self.dec_block_4(dec_x)
        if self.do_ds:
            x_ds_4 = self.out_4(x)
        del x_res_4, x_up_4

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3 
        x = self.dec_block_3(dec_x)
        if self.do_ds:
            # dec_out_3 = (torch.max(torch.abs(x), dim=1)[0]).unsqueeze(1)
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2 
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            # dec_out_2 = (torch.max(torch.abs(x), dim=1)[0]).unsqueeze(1)
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1 
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            # dec_out_1 = (torch.max(torch.abs(x), dim=1)[0]).unsqueeze(1)
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1
        
        # dec_out = (torch.max(torch.abs(x), dim=1)[0]).unsqueeze(1)
        # dec_out = (torch.max(x, dim=1)[0]).unsqueeze(1)
        dec_out = x.mean(dim=1, keepdim=True)

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0 
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.training:
            if self.do_ds:
                # return [dec_out, dec_out_1, dec_out_2, dec_out_3], [x, x_ds_1, x_ds_2, x_ds_3], loss_tp
                return dec_out, [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4], loss_tp
            else: 
                return dec_out, x, loss_tp
        else:
            if self.do_ds:
                return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
            else:
                return x

if __name__ == '__main__':
    model = PSPSeg_L(in_channels=1,
                    out_channels=3,
                    redundant_module=PRM,
                    n_channels=[16, 32, 64, 128, 256, 320],
                    block_counts=[1, 1, 1, 1, 1, 1],
                    exp_r=2,
                    para_num=7,
                    deep_supervision=True)
    # model = PSPSeg_L(in_channels=1,
    #                 out_channels=2,
    #                 redundant_module=PRM,
    #                 n_channels=[32, 64, 128, 256, 320, 320],
    #                 block_counts=[2,2,2,2,2,2],
    #                 exp_r=2,
    #                 para_num=7,
    #                 deep_supervision=True)
    # model.load_pretrained_ckpt('/media/userdisk0/lhli/nnUNet-master/pretrained_model/PSPSeg_l_ep1k.model')
    print(sum(p.numel() for p in model.parameters()))
    # print(sum(p.numel() for p in model.up_2.parameters()))
    # print(sum(p.numel() for p in model.up_1.parameters()))
    # print(sum(p.numel() for p in model.up_0.parameters()))
