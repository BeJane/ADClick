from isegm.data.datasets.mvtec import MvtecDataset
from isegm.data.datasets.mvtec_cls_prompt import Mvtec_ClsPrompt_Dataset
from isegm.data.datasets.mvtec_prompt import Mvtec_Prompt_Dataset
from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss
from isegm.utils.serialization import load_torch_checkpoint

MODEL_NAME = 'mvtec_zero_conv_clsprompt_plainvit'


def main(cfg):
    cfg.residual_stride = 4
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (448, 448)
    model_cfg.num_max_points = 24

    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(16,16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )
    residual_backbone_params = dict(
        img_size=(model_cfg.crop_size[0]//cfg.residual_stride,model_cfg.crop_size[1]//cfg.residual_stride),
        stride=cfg.residual_stride,
        use_zero_conv=True,
        in_chans=512,
        window_size=8,
        patch_size=1, embed_dim=512, depths=[2],
        num_heads=[16])

    neck_params = dict(
        in_dim = 768,
        out_dims = [128, 256, 512, 1024],
    )

    head_params = dict(
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        channels={'x1':256, 'x2': 128, 'x4': 64}[cfg.upsample]
    )

    model = PlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        residual_backbone_params=residual_backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
        prompt_mode='cls_prompt'
    )
    # Load image backbone weight
    ckpt = load_torch_checkpoint(cfg.IMAGENET_PRETRAINED_MODELS.SIMPLE_CLICK, map_location='cpu')
    msg = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(msg)
    model.to(cfg.device)
    # freeze language module
    for param in model.residual_backbone.residual_language.parameters():
        param.requires_grad = False
    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size

    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25),residual_stride=cfg.residual_stride),
        # Flip(),
        # RandomRotate90(),
        # ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
        #                  rotate_limit=(-3, 3), border_mode=0, p=0.75),
        TriPadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0,residual_stride=cfg.residual_stride),
        TriRandomCrop(*crop_size,residual_stride=cfg.residual_stride),
        # RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        # RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        TriPadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        TriRandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)
    trainset = Mvtec_ClsPrompt_Dataset(
        cfg.MVTEC_PATH,
        cfg.category,
        split='train',
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=3200,
    )

    valset = MvtecDataset(
        cfg.MVTEC_PATH,
        cfg.category,
        split='test',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=200
    )

    optimizer_params = {
        'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[50, 55], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        layerwise_decay=cfg.layerwise_decay,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=1,
                        image_dump_interval=300,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=10, validation=False)
