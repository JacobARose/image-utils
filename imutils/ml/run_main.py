"""

imutils/ml/run_main.py

python run_main.py train.pl_trainer.fast_dev_run=True

python run_main.py \
	+train.pl_trainer.limit_train_batches=1 \
	+train.pl_trainer.limit_val_batches=1 \
	+train.pl_trainer.limit_test_batches=1 \
	+train.pl_trainer.max_epochs=2 \
	data.datamodule.batch_size=16 \
	train.callbacks.class_counts_stats=null \
	train.callbacks.image_stats_accumulator=null
	
python run_main.py \
	+train.pl_trainer.limit_train_batches=5 \
	+train.pl_trainer.limit_val_batches=5 \
	+train.pl_trainer.limit_test_batches=5 \
	+train.pl_trainer.max_epochs=3 \
	data.datamodule.batch_size=16 \
	train.callbacks.class_counts_stats=null \
	train.callbacks.image_stats_accumulator=null


python run_main.py \
	train.pl_trainer.gpus=2 \
	+train.pl_trainer.accelerator="ddp" \
	+train.pl_trainer.max_epochs=30 \
	data.datamodule.batch_size=64 \
	train.callbacks.class_counts_stats=null \
	train.callbacks.image_stats_accumulator=null

(6 AM 2022-03-21)
- Fixing accelerator, strategy, and devices arguments
python run_main.py \
	train.pl_trainer.devices=2 \
	train.pl_trainer.accelerator="gpu" \
	+train.pl_trainer.strategy="ddp" \
	+train.pl_trainer.max_epochs=30 \
	data.datamodule.batch_size=64 \
	train.callbacks.class_counts_stats=null \
	train.callbacks.image_stats_accumulator=null
	

(6:20 AM 2022-03-21)
- Scaling down to 1 gpu to debug weird miscalculation of num batches in an epoch
- Updated config to remove the need to nullify extraneous callbacks w/ every cmdline execution

python run_main.py \
	train.pl_trainer.devices=1 \
	train.pl_trainer.accelerator="gpu" \
	+train.pl_trainer.max_epochs=30 \
	data.datamodule.batch_size=64
	
############################

(8:30 AM 2022-03-21)
- Staying on 1 gpu to see if speed improves at all with higher batch size, since current GPU memory use is low (<10%)
	- [x] Increasing batch_size from 64->128
		- Appears to have taken ~5 mins to actually begin fitting.
		- p3 is being used a bit more than before, but still should not be enough to cause any conflicts here.
		- However, as training is still beginning (batches 0-30) - loading speed appears to be ~3 s/batch
- Removing lr_scheduler to reduce complexity while debugging other things & developing baselines,
	- [] I will have to test it out at some point down the line.
- [x] Adding more logging guarantees to validate # of trainable parameters.
	- This took me like an hour, considering the issues with hydra resolving the wrong data type.
	- Set to log a textual model summary in the model hook `on_fit_start`
		- This confirmed that the backbone is indeed failing to properly freeze, currently trainable_params=57 million=100%.
- [x] Adding simple pl profiler.

export CUDA_VISIBLE_DEVICES=6; python run_main.py \
	train.pl_trainer.devices=1 \
	train.pl_trainer.accelerator="gpu" \
	+train.pl_trainer.max_epochs=30 \
	+train.pl_trainer.profiler="simple" \
	optim.use_lr_scheduler=False \
	data.datamodule.batch_size=128


#############################


(9:40 AM AM 2022-03-21)
- Fixing the freezing/unfreezing issue, on 1-gpu until then. Next, back to debugging multi-gpu.
- Took ~5 mins to start training again w/ bsz=128.

export CUDA_VISIBLE_DEVICES=7; python run_main.py \
	train.pl_trainer.devices=1 \
	train.pl_trainer.accelerator="gpu" \
	+train.pl_trainer.max_epochs=30 \
	+train.pl_trainer.profiler="simple" \
	optim.use_lr_scheduler=False \
	data.datamodule.batch_size=128

(10:10 AM AM 2022-03-21)
- Model.on_train_start didnt get called for some reason in the 9:40 AM run, plus it was going really slow, so I restarted it & lowered the batch size back down to 64.
- Took ~5 mins to start training again w/ bsz=64.
- Observation: It appears my freezing function was accidentally using layer.enable_grad rather than the correct layer.requires_grad attribute

- [SOLVED] - at 11:00 AM 2022-03-21

export CUDA_VISIBLE_DEVICES=7; python run_main.py \
	train.pl_trainer.devices=1 \
	train.pl_trainer.accelerator="gpu" \
	+train.pl_trainer.max_epochs=30 \
	+train.pl_trainer.profiler="simple" \
	optim.use_lr_scheduler=False \
	data.datamodule.batch_size=64



(11:15 AM 2022-03-21)
- Setting random__flips=False to reduce compute
- Moving back to 2-gpus and strategy=DDP, batch_size=128
- freezing full backbone now


- Result (11:57 AM): [OOM ERROR] -- Crashed in the middle of Fossil Sync meeting
	- wandb shows process gpu memory increasing linearly until the crash, indicating a likely MEMORY LEAK.
	- Proposed solution: Refactor the steps to not return full image vectors at every batch.

export CUDA_VISIBLE_DEVICES='5,7'; python run_main.py \
	train.pl_trainer.devices=2 \
	train.pl_trainer.accelerator="gpu" \
	+train.pl_trainer.strategy="ddp" \
	+train.pl_trainer.max_epochs=30 \
	+train.pl_trainer.profiler="simple" \
	optim.use_lr_scheduler=False \
	data.datamodule.batch_size=128

################


(12:00 PM AM 2022-03-21)
Goals:
	- 1. Trying to identify and remove the possible memory leak
	- 2. Trying to debug why DDP isnt working on both GPUs
Actions:
	- 1. Refactoring step to not return x at every batch.
- Keeping to 2-gpus and strategy=DDP, batch_size=128
- freezing full backbone

- (12:15 PM) -- Removed x from step returns to see if the memory leak disappears
	- Also printing available GPU devices between model creations & DDP.


export CUDA_VISIBLE_DEVICES='0,4'; python run_main.py \
		train.pl_trainer.devices=2 \
		train.pl_trainer.accelerator="gpu" \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=64


#################################################


(1:10 PM AM 2022-03-21)
Goals:
	- 1. Trying to specify GPUs without CUDA_VISIBLE_DEVICES
	- 2. Trying to debug why DDP isnt working on both GPUs
Actions:
	- 1. specifying devices=[0,4]
	
- Had to enter this command first:
unset CUDA_VISIBLE_DEVICES

python run_main.py \
		train.pl_trainer.devices=[0,4] \
		train.pl_trainer.accelerator="gpu" \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=128

##################################


python run_main.py \
		train.pl_trainer.devices=[0,4] \
		train.pl_trainer.accelerator="gpu" \
		data.datamodule.num_workers=1 \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=128


python run_main.py \
		train.pl_trainer.devices=[0,4] \
		train.pl_trainer.accelerator="gpu" \
		data.datamodule.num_workers=0 \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=128
		
python run_main.py \
		train.pl_trainer.devices=[0,4] \
		train.pl_trainer.accelerator="gpu" \
		data.datamodule.num_workers=8 \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=128
		
################################


(~2:40 PM)

python run_main.py \
		train.pl_trainer.devices=[0,4] \
		train.pl_trainer.accelerator="gpu" \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=128 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf


####################################


(~3 PM)

(~2:40 PM)

python run_main.py \
		train.pl_trainer.devices=[0,4,5,6] \
		train.pl_trainer.accelerator="gpu" \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=32 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf

################

(~3:40 PM) -- Removed gradient_clip_val = 10,000,000 from default trainer cfg.

python run_main.py \
		train.pl_trainer.devices=[0,4,5,6] \
		train.pl_trainer.accelerator="gpu" \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=32 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf
		
		
python run_main.py \
		train.pl_trainer.devices=[0,4,5,6] \
		train.pl_trainer.accelerator="gpu" \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=24 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf


#############################


python run_main.py \
		train.pl_trainer.devices=[0,4,5,6] \
		train.pl_trainer.accelerator="gpu" \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=24 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf


###################

python run_main.py \
		train.pl_trainer.devices=[0,4] \
		train.pl_trainer.accelerator="gpu" \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.strategy="ddp" \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=32 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf \
		+overfit_batches=10


#############################################

(11:30 PM 2022-03-22)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=32 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf

#############################################

(12:40 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		+train.pl_trainer.limit_train_batches=0.5 \
		+train.pl_trainer.limit_val_batches=0.5 \
		+train.pl_trainer.limit_test_batches=0.5 \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=48 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(1:25 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		+train.pl_trainer.limit_train_batches=0.5 \
		+train.pl_trainer.limit_val_batches=0.5 \
		+train.pl_trainer.limit_test_batches=0.5 \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=64 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(2:45 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		+train.pl_trainer.limit_train_batches=0.5 \
		+train.pl_trainer.limit_val_batches=0.5 \
		+train.pl_trainer.limit_test_batches=0.5 \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=96 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf


###############################################

(3:15 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		+train.pl_trainer.limit_train_batches=0.5 \
		+train.pl_trainer.limit_val_batches=0.5 \
		+train.pl_trainer.limit_test_batches=0.5 \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=128 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(4:30 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		+train.pl_trainer.limit_train_batches=0.5 \
		+train.pl_trainer.limit_val_batches=0.5 \
		+train.pl_trainer.limit_test_batches=0.5 \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=144 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(5:45 AM 2022-03-23) Experiment #7
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		+train.pl_trainer.limit_train_batches=0.5 \
		+train.pl_trainer.limit_val_batches=0.5 \
		+train.pl_trainer.limit_test_batches=0.5 \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="simple" \
		train.pl_trainer.accumulate_grad_batches=2 \
		optim.use_lr_scheduler=False \
		data.datamodule.batch_size=144 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(8:50 AM 2022-03-23) Experiment #8
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		+train.pl_trainer.limit_train_batches=0.1 \
		+train.pl_trainer.limit_val_batches=0 \
		+train.pl_trainer.limit_test_batches=0 \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="advanced" \
		train.pl_trainer.accumulate_grad_batches=2 \
		optim.use_lr_scheduler=False \
		optim.optimizer.lr=1e-2 \
		data.datamodule.batch_size=128 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf


###############################################

(11:20 AM 2022-03-23) Experiment #9
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		+train.pl_trainer.limit_train_batches=0.01 \
		+train.pl_trainer.limit_val_batches=0 \
		+train.pl_trainer.limit_test_batches=0 \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		train.freeze_backbone_up_to=-3 \
		+train.pl_trainer.profiler="advanced" \
		train.pl_trainer.accumulate_grad_batches=1 \
		optim.use_lr_scheduler=False \
		optim.optimizer.lr=1e-2 \
		data.datamodule.batch_size=128 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(11:45 AM 2022-03-23) Experiment #10
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		+train.pl_trainer.limit_train_batches=0.05 \
		+train.pl_trainer.limit_val_batches=0 \
		+train.pl_trainer.limit_test_batches=0 \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		train.freeze_backbone_up_to=-3 \
		+train.pl_trainer.overfit_batches=0.05 \
		+train.pl_trainer.track_grad_norm=2 \
		+train.pl_trainer.profiler="advanced" \
		train.pl_trainer.accumulate_grad_batches=1 \
		optim.use_lr_scheduler=False \
		optim.optimizer.lr=1e-2 \
		data.datamodule.batch_size=128 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf
		
###############################################

(12:25 PM 2022-03-23) Experiment #11
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		train.freeze_backbone_up_to=-4 \
		+train.pl_trainer.profiler="advanced" \
		train.pl_trainer.accumulate_grad_batches=1 \
		optim.use_lr_scheduler=False \
		optim.optimizer.lr=0.5e-3 \
		data.datamodule.batch_size=128 \
		hp.resolution=448 \
		aug@data.datamodule.transform_cfg=medium_image_aug_conf


###############################################

(3:00 PM 2022-03-23) Experiment #12
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
		train.pl_trainer.devices=2 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		train.freeze_backbone_up_to=-4 \
		+train.pl_trainer.profiler="advanced" \
		train.pl_trainer.accumulate_grad_batches=1 \
		optim.use_lr_scheduler=False \
		optim.optimizer.lr=1e-2 \
		data.datamodule.batch_size=128 \
		hp.preprocess_size=256 \
		hp.resolution=224 \
		aug@data.datamodule.transform_cfg=default_image_aug_conf


###############################################

(3:00 PM 2022-03-23) Experiment #13
export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
		train.pl_trainer.devices=4 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		train.freeze_backbone_up_to=-4 \
		+train.pl_trainer.profiler="advanced" \
		train.pl_trainer.accumulate_grad_batches=1 \
		optim.use_lr_scheduler=False \
		optim.optimizer.lr=2e-2 \
		data.datamodule.batch_size=128 \
		hp.preprocess_size=256 \
		hp.resolution=224 \
		aug@data.datamodule.transform_cfg=default_image_aug_conf



###############################################
- OOM crash after bsz=128 and accumulate_grad_batches=1, so trying 1/2 the bsz and 2x the accumulate
(5:00 PM 2022-03-23) Experiment #14
export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
		train.pl_trainer.devices=4 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		train.freeze_backbone_up_to=-4 \
		+train.pl_trainer.profiler="advanced" \
		train.pl_trainer.accumulate_grad_batches=2 \
		optim.use_lr_scheduler=False \
		optim.optimizer.lr=1e-3 \
		optim.optimizer.weight_decay=1e-6 \
		data.datamodule.batch_size=64 \
		hp.preprocess_size=256 \
		hp.resolution=224 \
		aug@data.datamodule.transform_cfg=default_image_aug_conf \
		model_cfg.backbone.name=resnext50_32x4d \
		model_cfg.backbone.pretrained=false \
		model_cfg.backbone.freeze_backbone=false \
		train.freeze_backbone_up_to=0 \
		train.freeze_backbone=false



###############################################
(4:20 AM 2022-03-24) Experiment #15
- Restored model checkpointing to code after #14's success.

export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
		train.pl_trainer.devices=4 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="advanced" \
		train.pl_trainer.accumulate_grad_batches=2 \
		optim.use_lr_scheduler=False \
		optim.optimizer.lr=5e-3 \
		optim.optimizer.weight_decay=1e-5 \
		data.datamodule.batch_size=64 \
		aug@data.datamodule.transform_cfg=default_image_aug_conf \
		hp.preprocess_size=256 \
		hp.resolution=224 \
		model_cfg.backbone.name=resnext50_32x4d \
		model_cfg.backbone.pretrained=false \
		model_cfg.backbone.freeze_backbone=false \
		train.freeze_backbone_up_to=0 \
		train.freeze_backbone=false


###############################################
(7:35 AM 2022-03-24) Experiment #15b
- Performance is slightly worse in Exp #15 compared to Experiment #14 within the first 2 epochs -> Attempting to reduce lr down from 5e-3 (Exp#15) -> 2e-3, vs. 1e-3 (Exp#14)

export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
		train.pl_trainer.devices=4 \
		data.datamodule.num_workers=4 \
		+train.pl_trainer.max_epochs=30 \
		+train.pl_trainer.profiler="advanced" \
		train.pl_trainer.accumulate_grad_batches=2 \
		optim.use_lr_scheduler=False \
		optim.optimizer.lr=2e-3 \
		optim.optimizer.weight_decay=1e-5 \
		data.datamodule.batch_size=64 \
		aug@data.datamodule.transform_cfg=default_image_aug_conf \
		hp.preprocess_size=256 \
		hp.resolution=224 \
		model_cfg.backbone.name=resnext50_32x4d \
		model_cfg.backbone.pretrained=false \
		model_cfg.backbone.freeze_backbone=false \
		train.freeze_backbone_up_to=0 \
		train.freeze_backbone=false


###############################################
(11:45 AM 2022-03-25) Experiment #16

- Increasing weight_decay from 1e-5->2e-05
- lr = 2e-3
- LinearWarmupCosineAnnealingLR
		pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
		warmup_epochs: 3
		max_epochs: ${hp.max_epochs}
		warmup_start_lr: 1e-04
		eta_min: 1e-06
		last_epoch: -1


export CUDA_VISIBLE_DEVICES=4,5,6,7; python run_main.py \
train.pl_trainer.devices=4 \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=30 \
+train.pl_trainer.profiler="advanced" \
train.pl_trainer.accumulate_grad_batches=2 \
optim.optimizer.lr=2e-3 \
optim.optimizer.weight_decay=2e-5 \
data.datamodule.batch_size=64 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.preprocess_size=256 \
hp.resolution=224 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=false \
model_cfg.backbone.freeze_backbone=false \
train.freeze_backbone_up_to=0 \
train.freeze_backbone=false


###############################################
(3:00 PM 2022-03-25) Experiment #16b

- **Decreasing** weight_decay back from 2e-5->5e-06
- lr = 2e-3
- LinearWarmupCosineAnnealingLR
		pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
		warmup_epochs: 3
		max_epochs: ${hp.max_epochs}
		warmup_start_lr: 1e-04
		eta_min: 1e-06
		last_epoch: -1


export CUDA_VISIBLE_DEVICES=4,5,6,7; python run_main.py \
train.pl_trainer.devices=4 \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=30 \
+train.pl_trainer.profiler="advanced" \
train.pl_trainer.accumulate_grad_batches=2 \
optim.optimizer.lr=2e-3 \
optim.optimizer.weight_decay=5e-6 \
data.datamodule.batch_size=64 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.preprocess_size=256 \
hp.resolution=224 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=false \
model_cfg.backbone.freeze_backbone=false \
train.freeze_backbone_up_to=0 \
train.freeze_backbone=false



###############################################
(9:30 PM 2022-03-25) Experiment #16c

- DEBUGGING crash at end of Experiment #16b validation stage of epoch 0

export CUDA_VISIBLE_DEVICES=7; python run_main.py \
train.pl_trainer.devices=1 \
data.datamodule.num_workers=1 \
train.pl_trainer.fast_dev_run=true \
train.pl_trainer.max_epochs=2 \
+train.pl_trainer.profiler="advanced" \
train.pl_trainer.accumulate_grad_batches=2 \
optim.optimizer.lr=2e-3 \
optim.optimizer.weight_decay=5e-6 \
data.datamodule.batch_size=64 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.preprocess_size=256 \
hp.resolution=224 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=false \
model_cfg.backbone.freeze_backbone=false \
train.freeze_backbone_up_to=0 \
train.freeze_backbone=false

###################
###################

# (11:15 PM 2022-03-25) Experiment #16d

export CUDA_VISIBLE_DEVICES=7; python run_main.py \
train.pl_trainer.devices=1 \
data.datamodule.num_workers=1 \
+train.pl_trainer.limit_train_batches=5 \
+train.pl_trainer.limit_val_batches=5 \
train.pl_trainer.max_epochs=4 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1 \
optim.optimizer.lr=2e-3 \
optim.optimizer.weight_decay=5e-6 \
data.datamodule.batch_size=16 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.preprocess_size=256 \
hp.resolution=224 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=false \
model_cfg.backbone.freeze_backbone=false \
train.freeze_backbone_up_to=0 \
train.freeze_backbone=false


###################

# (11:15 PM 2022-03-25) Experiment #16e

#16d worked without error after tediously passing through using pdb. 
- Trying the same with num_workers set from 1->4
- Set limit_train_batches from 5->10
- Set limit_val_batches from 5->10


--

- Might have been caused by accidentally using a strategy='ddp' by default even though devices=1?
(11:28 PM) -> SUCCESS: It was the problem caused by using "ddp" on 1 gpu!!

export CUDA_VISIBLE_DEVICES=7; python run_main.py \
train.pl_trainer.devices=1 \
train.pl_trainer.strategy=null \
data.datamodule.num_workers=4 \
+train.pl_trainer.limit_train_batches=10 \
+train.pl_trainer.limit_val_batches=10 \
train.pl_trainer.max_epochs=4 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1 \
optim.optimizer.lr=2e-3 \
optim.optimizer.weight_decay=5e-6 \
data.datamodule.batch_size=16 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.preprocess_size=256 \
hp.resolution=224 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=false \
model_cfg.backbone.freeze_backbone=false \
train.freeze_backbone_up_to=0 \
train.freeze_backbone=false


################################

###############################################
(11:33 PM 2022-03-25) Experiment #16f

- Repeating settings from Experiment #16b after finding the code bug.
	- Only change is setting optim.lr_scheduler.warmup_start_lr from 1e-04 -> 1e-03

- **Decreasing** weight_decay back from 2e-5->5e-06
- lr = 2e-3
- LinearWarmupCosineAnnealingLR
		pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
		warmup_epochs: 3
		max_epochs: ${hp.max_epochs}
		warmup_start_lr: 1e-04
		eta_min: 1e-06
		last_epoch: -1


export CUDA_VISIBLE_DEVICES=4,5,6,7; python run_main.py \
train.pl_trainer.devices=4 \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=30 \
+train.pl_trainer.profiler="advanced" \
train.pl_trainer.accumulate_grad_batches=2 \
optim.optimizer.lr=2e-3 \
optim.lr_scheduler.warmup_start_lr=1e-03 \
optim.optimizer.weight_decay=5e-6 \
data.datamodule.batch_size=64 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.preprocess_size=256 \
hp.resolution=224 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=false \
model_cfg.backbone.freeze_backbone=false \
train.freeze_backbone_up_to=0 \
train.freeze_backbone=false



###############################################
(2:50 AM 2022-03-26) Experiment #16g
- launched @ 3:10 AM
- Repeating settings from Experiment #16b after finding the code bug.
	- Only changes are:
		- setting optim.lr_scheduler.warmup_start_lr from 1e-04 -> 1e-03
		- Removing preprocess_size in order to (hopefully) only resize images once to the output size/resolution
		- Switching model=resnext50_32x4d -> model=resnet18
		- Switching pretrained=false -> pretrained=true
		- Switching devices=2
- **Decreasing** weight_decay back from 2e-5->5e-06
- lr = 2e-3
- LinearWarmupCosineAnnealingLR
		pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
		warmup_epochs: 3
		max_epochs: ${hp.max_epochs}
		warmup_start_lr: 1e-04
		eta_min: 1e-06
		last_epoch: -1


export CUDA_VISIBLE_DEVICES=4,5; python run_main.py \
optim.optimizer.lr=2e-3 \
optim.lr_scheduler.warmup_start_lr=1e-03 \
optim.optimizer.weight_decay=5e-6 \
data.datamodule.batch_size=32 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.resolution=224 \
model_cfg.backbone.name=resnet18 \
model_cfg.backbone.pretrained=true \
model_cfg.backbone.freeze_backbone=false \
train.freeze_backbone_up_to=0 \
train.freeze_backbone=false \
train.pl_trainer.devices=2 \
data.datamodule.num_workers=4 \
train.pl_trainer.strategy=null \
data.datamodule.num_workers=4 \
+train.pl_trainer.limit_train_batches=10 \
+train.pl_trainer.limit_val_batches=10 \
train.pl_trainer.max_epochs=4 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1



###############################################
(3:30 AM 2022-03-26) Experiment #16h

- 16g crashed with 2-gpus, now going back to 1.

- Repeating settings from Experiment #16b after finding the code bug.
	- Only change is setting optim.lr_scheduler.warmup_start_lr from 1e-04 -> 1e-03
	- Removing preprocess_size in order to (hopefully) only resize images once to the output size/resolution
	- Switching model=resnext50_32x4d -> model=resnet18
	- Switching pretrained=false -> pretrained=true
	- Switching from devices=2 -> devices=1
- **Decreasing** weight_decay back from 2e-5->5e-06
- lr = 2e-3
- LinearWarmupCosineAnnealingLR
		pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
		warmup_epochs: 3
		max_epochs: ${hp.max_epochs}
		warmup_start_lr: 1e-04
		eta_min: 1e-06
		last_epoch: -1


export CUDA_VISIBLE_DEVICES=4,5; python run_main.py \
optim.optimizer.lr=2e-3 \
optim.lr_scheduler.warmup_start_lr=1e-03 \
optim.optimizer.weight_decay=5e-6 \
data.datamodule.batch_size=16 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.resolution=224 \
model_cfg.backbone.name=resnet18 \
model_cfg.backbone.pretrained=true \
model_cfg.backbone.freeze_backbone=false \
train.freeze_backbone_up_to=0 \
train.freeze_backbone=false \
train.pl_trainer.devices=2 \
data.datamodule.num_workers=4 \
+train.pl_trainer.limit_train_batches=25 \
+train.pl_trainer.limit_val_batches=25 \
train.pl_trainer.max_epochs=2 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1

# train.pl_trainer.strategy=null \



###############################################
(4:30 AM 2022-03-27) Experiment #16i
	- finished ~5:30 AM
	- [Result: PARTIAL SUCCESS] — Managed to get ddp_spawn to make it past epoch 0 validation into epoch 1+

- 16g crashed with 2-gpus, now going back to 1.
- 16h finished successfully with 1-gpu after compromising on a validation epoch logging issue (~5:30 AM 2022-03-26)
- Now, #16i will try with 2-gpus again


Notable:

- weight_decay=5e-06
- lr = 2e-3
- Adding experiment name to wandb config that corresponds to my notes:
	`core.name="Experiment #16h (2022-03-27)"`
- Added second `--config-name=dev_conf`
- Overriding `callbacks=null`
- Moved `train.freeze_backbone_up_to` and `train.freeze_backbone` to be interpolations of a user-specified value under `hp` (for `hyper-parameters`)
- Changed default trainer strategy = `ddp_spawn`.
- LinearWarmupCosineAnnealingLR
		pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
		warmup_epochs: 3
		max_epochs: ${hp.max_epochs}
		warmup_start_lr: 1e-04
		eta_min: 1e-06


export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
--config-name=dev_conf \
'core.name="Experiment #16i (2022-03-27)"' \
optim.optimizer.lr=2e-3 \
optim.lr_scheduler.warmup_start_lr=1e-03 \
optim.optimizer.weight_decay=5e-6 \
data.datamodule.batch_size=32 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.resolution=224 \
model_cfg.backbone.name=resnet18 \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=2 \
data.datamodule.num_workers=4 \
+train.pl_trainer.limit_train_batches=25 \
+train.pl_trainer.limit_val_batches=25 \
train.pl_trainer.max_epochs=2 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1





###############################################
(5:50 AM 2022-03-27) Experiment #16j
	- finished ~6 AM
	- [Result: SUCCESS] - Managed to get ddp to make it past epoch 0 validation into epoch 1+, all other settings identical to Experiment #16i

- 16g crashed with 2-gpus, now going back to 1.
- 16h finished successfully with 1-gpu after compromising on a validation epoch logging issue (~5:30 AM 2022-03-26)
- 16i got 2-gpus working again by removing some necessary logging & callback code + switching to ddp_spawn.
- Now, 16j will switch back to ddp, otherwise settingsare identical to Experiment #16i



Notable:
	- Switching strategy back from `ddp_spawn` to `ddp`
	- devices still equals 2.

export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
--config-name=dev_conf \
'core.name="Experiment #16j (2022-03-27)"' \
optim.optimizer.lr=2e-3 \
optim.lr_scheduler.warmup_start_lr=1e-03 \
optim.optimizer.weight_decay=5e-6 \
data.datamodule.batch_size=32 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.resolution=224 \
model_cfg.backbone.name=resnet18 \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=2 \
data.datamodule.num_workers=4 \
+train.pl_trainer.limit_train_batches=25 \
+train.pl_trainer.limit_val_batches=25 \
train.pl_trainer.max_epochs=2 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1



###############################################
(6:10 AM 2022-03-27) Experiment #16k
	- finished ~6:35 AM
	[Task: Restore metrics logging (on DDP)]
	— [Result: SUCCESS] — Train and validation metrics all logged to wandb for 2 (shortened) epochs.
	

- 16g crashed with 2-gpus, now going back to 1.
- 16h finished successfully with 1-gpu after compromising on a validation epoch logging issue (~5:30 AM 2022-03-26)
- 16i got 2-gpus working again by removing some necessary logging & callback code + switching to ddp_spawn.
- 16j switched back to ddp, otherwise settings were identical to Experiment #16i
- 16k will restore metric logging to validation loop


Notable:
	- strategy still set to `ddp`
	- devices still equals 2.
	- Restored metric logging functionality to validation steps
	- moved batch_size specification from data.datamodule.batch_size -> to -> hp.batch_size


export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
--config-name=dev_conf \
'core.name="Experiment #16k (2022-03-27)"' \
optim.optimizer.lr=2e-3 \
optim.lr_scheduler.warmup_start_lr=1e-03 \
optim.optimizer.weight_decay=5e-6 \
hp.batch_size=32 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.resolution=224 \
model_cfg.backbone.name=resnet18 \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=2 \
data.datamodule.num_workers=4 \
+train.pl_trainer.limit_train_batches=25 \
+train.pl_trainer.limit_val_batches=25 \
train.pl_trainer.max_epochs=2 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1




###############################################
(8:30 AM 2022-03-27) Experiment #16l
	- finished ~8:50 AM
	[Task: Restore base callbacks logging (on DDP)]
	[Result: PARTIAL SUCCESS] — Restored the base_callbacks.yaml (not yet the full default_callbacks.yaml). Callbacks didn’t trigger any major errors, but experiment logging is still not as clean as I’d like 
	

- 16g crashed with 2-gpus, now going back to 1.
- 16h finished successfully with 1-gpu after compromising on a validation epoch logging issue (~5:30 AM 2022-03-26)
- 16i got 2-gpus working again by removing some necessary logging & callback code + switching to ddp_spawn.
- 16j switched back to ddp, otherwise settings were identical to Experiment #16i
- 16k restored metric logging to validation loop
- 16l will restore base_callbacks.yaml to training

base_callbacks:
	- (Omitted for now) progress_bar
	- lr_monitor
	- early_stopping
	- model_checkpoint

Notable:
	- strategy still set to `ddp`
	- devices still equals 2.
	- Restored metric logging functionality to validation steps
	- moved batch_size specification from data.datamodule.batch_size -> to -> hp.batch_size
	- callbacks@train.callbacks: base_callbacks #default


export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
'core.name="Experiment #16l (2022-03-27)"' \
optim.optimizer.lr=2e-3 \
optim.lr_scheduler.warmup_start_lr=1e-03 \
optim.optimizer.weight_decay=5e-6 \
hp.batch_size=32 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.resolution=224 \
model_cfg.backbone.name=resnet18 \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=2 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=0 \
+train.pl_trainer.limit_train_batches=25 \
+train.pl_trainer.limit_val_batches=25 \
train.pl_trainer.max_epochs=2 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1




###############################################
(11:45 PM 2022-03-27) Experiment #16m
	- finished ~2:15 AM 2022-03-28
	[Task: Restore default callbacks logging (on DDP)]
	[Result: SUCCESS x3] — 
		(1) Restored the default_callbacks.yaml. 
		(2) Fixed the problems with recursive experiment dirs. 
		(3) Solidified my implementation of wandb artifact saving for model checkpoint management.

Note:
	- Having to do many short runs on this one due to issues being caused by wandb.Artifacts trying to save model checkpoints on more than just the rank=0 device, and finding that they cannot see the checkpoint files for some reason.

- 16g crashed with 2-gpus, now going back to 1.
- 16h finished successfully with 1-gpu after compromising on a validation epoch logging issue (~5:30 AM 2022-03-26)
- 16i got 2-gpus working again by removing some necessary logging & callback code + switching to ddp_spawn.
- 16j switched back to ddp, otherwise settings were identical to Experiment #16i
- 16k restored metric logging to validation loop
- 16l restored base_callbacks.yaml to training
- 16m will restore default_callbacks.yaml to training


base_callbacks:
	- (Omitted for now) progress_bar
	- lr_monitor
	- early_stopping
	- model_checkpoint

wandb_callbacks:
	- watch_model_with_wandb
	- uploadcheckpointsasartifact
	- module_data_monitor


Notable:
	- strategy still set to `ddp`
	- devices still equals 2.
	- Restored default callbacks
	- callbacks@train.callbacks: default_callbacks #default
	
	- Added the following value to cfg.base, which is then referenced by hydra to always use an absolute path instead of a relative path:
		- experiments_root_dir: "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/2022/herbarium2022"
	- Altered model_checkpoint filename:
		- from: 
			filename: '{epoch:02d}-{val_loss:.3f}-{${..kwargs.monitor.metric}:.3f}'
		- to:
			filename: '{epoch:02d}-{val_loss:.3f}-{${..kwargs.monitor.metric}:.3f}/model_weights'
	- Updated the artifact name & type fields to have slightly different defaults & to be easily configurable
	- Added @rank_zero_only wrapper for wandb model artifact uploading & for template_utils.finish() helper function.


export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
'core.name="Experiment #16m (2022-03-27)"' \
optim.optimizer.lr=2e-3 \
optim.lr_scheduler.warmup_start_lr=1e-03 \
optim.optimizer.weight_decay=5e-6 \
hp.batch_size=32 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.resolution=224 \
model_cfg.backbone.name=resnet18 \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=2 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=0 \
+train.pl_trainer.limit_train_batches=6 \
+train.pl_trainer.limit_val_batches=6 \
train.pl_trainer.max_epochs=1 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1

########



##########################
##########################
### **********************
##########################
### **********************
##########################
##########################


###############################################
###############################################
##########################
### **********************
# Adding new pretrain execution stage: lr_tune
### **********************
##########################

###############################################
(7:20 AM 2022-03-28) Experiment #17
	- (7:41 AM Monday) -> Finally got lr_tuner to make it past 0% on the progress bar
		- Forgot to pass non-default kwargs to lr_find function call:
				lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule, **tuner_args)
	- (8:10 AM) -> Set up the lr_tuner to either replace the main cfg.optim.optimizer.lr if using no scheduler, or replace cfg.optim.lr_scheduler.warmup_start_lr if using one.
		- However, it appears to have almost worked too well! I think I need to wrap the pretrain phase in @rank_zero_only
			- Despite accidentally running double the lr_tuner stages (1 on each GPU), trainer.fit was able to start successfully.
	- finished ~8:20 AM 2022-03-28
	[Task: Get MVP of lr_tuner up and running then launch full training.]
	[Result: **SUCCESS**] — lr_tuner is working & has an [EXPERIMENTAL] option to either update the main optimizer lr, or the scheduler's warmup_start_lr.


Notable:
	- strategy=`ddp`
	- devices=2

export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
'core.name="Experiment #17 (2022-03-28)"' \
optim.optimizer.lr=2e-3 \
optim.use_lr_scheduler=false \
optim.optimizer.weight_decay=5e-6 \
hp.batch_size=96 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.resolution=224 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=2 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1



###############################################
##########################
### **********************
# Implementing single script that correctly performs in 2 stages:
	1. pretrain execution stage: lr_tune
	2. 4-GPU DDP training with lr_scheduler
### **********************
##########################

###############################################
(8:20 AM 2022-03-28) Experiment #18
	[Task: Get 4-GPU full training launched with the pretrain stage WIP lr_tuner]
	- (9:21 AM) - Launched again after attempting workaround solution for DDP problems when using an lr_tuner stage to decide the config for all GPUs
	- (9:32 AM) - PARTIAL SUCCESS - But I had to restart training after batch ~150 or so due to mismatch between current config setup and the command line option I passed.
		- Launching again, but replacing `optim.optimizer.lr=2e-03` with `hp.lr=2e-03`, which automatically passes on to the former through interpolation.
	[Result: ] — 

Notable:
	- strategy=`ddp`
	- devices=4
	- lr_tuner stage sets the optim.lr_scheduler.warmup_start_lr
	- Added experimental learning rate scaling to new_lr*num_gpus (see pretrain.lr_tuner.run())


export CUDA_VISIBLE_DEVICES=4,5,6,7; python run_main.py \
'core.name="Experiment #18 (2022-03-28)"' \
optim.optimizer.weight_decay=5e-6 \
hp.batch_size=96 \
hp.lr=2e-3 \
aug@data.datamodule.transform_cfg=default_image_aug_conf \
hp.resolution=224 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1


#####################
Experiment #20:

Switching source images from res=960 to res=512 on-disk.

export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
'core.name="Experiment #20 (2022-04-01)"' \
optim.optimizer.weight_decay=5e-6 \
hp.batch_size=24 \
hp.lr=2e-3 \
data/datamodule@data=herbarium2022-res_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug_conf \
hp.preprocess_size=512 \
hp.resolution=448 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=2

###################
Experiment #21:

- Extending warmup epochs from 3 to 5
- Setting hp.preprocess_size=None

Observations:

 - Note: It looks like the lr_tuner did not result in an identical warmup_start_lr between Exp #20 and #21, indicating a potential replicability problem.
	 #20: warmup_start_lr = 1.208e-5 (About 437% the magnitude of #21)
	 #21: warmup_start_lr = 2.767e-6 (Only about 23% the magnitude of #20)


export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
'core.name="Experiment #21 (2022-04-01)"' \
hp.warmup_epochs=5 \
optim.optimizer.weight_decay=5e-6 \
hp.batch_size=24 \
hp.lr=2e-3 \
data/datamodule@data=herbarium2022-res_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug_conf \
hp.preprocess_size=None \
hp.resolution=448 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=2






"""


# import logging
# import os
# import shutil
# from pathlib import Path
# from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
from icecream import ic
# import omegaconf
import os
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl

from imutils.ml.utils.common import load_envs
from imutils.ml.utils import template_utils
logging = template_utils.get_logger(__file__)

# Set the cwd to the project root
# os.chdir(Path(__file__).parent.parent)

# Load environment variables
load_envs()



def init_cfg(cfg: DictConfig):
	
	if cfg.train.deterministic:
		pl.seed_everything(cfg.train.random_seed)

	if cfg.train.pl_trainer.fast_dev_run:
		hydra.utils.log.info(
			f"Debug mode <{cfg.train.pl_trainer.fast_dev_run}>. "
			f"Forcing debugger friendly configuration!"
		)
		# Debuggers don't like GPUs nor multiprocessing
		if cfg.train.callbacks.get('watch_model_with_wandb') is not None:
			del cfg.train.callbacks.watch_model_with_wandb
		if cfg.train.callbacks.get('uploadcheckpointsasartifact') is not None:
			del cfg.train.callbacks.uploadcheckpointsasartifact
		if cfg.train.callbacks.get('model_checkpoint') is not None:
			del cfg.train.callbacks.model_checkpoint
		# cfg.train.pl_trainer.gpus = 0
		cfg.data.datamodule.num_workers = 0

	cfg.run_output_dir = os.path.abspath(cfg.run_output_dir)
	
	return cfg




def run_pretrain(cfg: DictConfig) -> None:
	"""
	Generic pretrain loop

	:param cfg: run configuration, defined by Hydra in /conf
	"""
	
	import os
	
	cfg = init_cfg(cfg)

	hydra_dir = os.path.abspath(os.getcwd())
	print(f"Using hydra_dir: {hydra_dir}")
	hydra.utils.log.info(f"Before pretrain.lr_tuner value of lr: {cfg.optim.optimizer.lr}")	
	if cfg.execution_list.auto_lr_tune:
		from imutils.ml import pretrain
		cfg = pretrain.lr_tuner.run(cfg=cfg)
						   # datamodule=datamodule)
						   # model=model)
	return cfg



def train(cfg: DictConfig) -> None:
	"""
	Generic train loop

	:param cfg: run configuration, defined by Hydra in /conf
	"""
	
	from rich import print as pp
	# import torch
	from imutils.ml.utils.experiment_utils import configure_callbacks, configure_loggers, configure_trainer
	import imutils.ml.models.pl.classifier
	
	cfg = init_cfg(cfg)

	hydra_dir = os.path.abspath(os.getcwd())	
	print(f"Using hydra_dir: {hydra_dir}")

	
	if cfg.execution_list.model_fit:
		
		hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
		datamodule: pl.LightningDataModule = hydra.utils.instantiate(
			cfg.data.datamodule, _recursive_=False
		)
		datamodule.setup()
		hydra.utils.log.info(f"Instantiating <{cfg.model_cfg._target_}> before trainer.fit()")
		# model: pl.LightningModule = hydra.utils.instantiate(cfg.model, cfg=cfg, _recursive_=False)
		model = imutils.ml.models.pl.classifier.LitClassifier(cfg=cfg, #model_cfg=cfg.model_cfg,
															  loss=cfg.model_cfg.loss)

		ic(model.lr, cfg.hp.lr, cfg.optim.optimizer.lr)
		
		loggers = configure_loggers(cfg=cfg, model=model)
		# Instantiate the callbacks
		callbacks: List[pl.Callback] = configure_callbacks(cfg=cfg.train)	
		hydra.utils.log.info(f"Instantiating the Trainer")
		pp(OmegaConf.to_container(cfg.train.pl_trainer))	
		trainer = configure_trainer(cfg,
									callbacks=callbacks,
									logger=loggers)

	# num_samples = len(datamodule.train_dataset)
		num_classes = cfg.model_cfg.head.num_classes
		batch_size = datamodule.batch_size #["train"]
		hydra.utils.log.info("Starting training with {} classes and batches of {} images".format(
			num_classes,
			batch_size))
		trainer.fit(model=model, datamodule=datamodule)


	template_utils.finish(
		config=cfg,
		logger=loggers,
		callbacks=callbacks)

	# if args.train:
	#	 trainer.fit(model, dm)
	# if args.test:
	#	 ckpt_path = (
	#		 checkpoint_callback.best_model_path if args.train else cfg.model.checkpoint
	#	 )
	#	 trainer.test(model=model, datamodule=dm)

	# print(f"Skipping testing for now, must run predict on unlabeled test set")
	# hydra.utils.log.info(f"Starting testing!")
	# trainer.test(model=model, datamodule=datamodule)
	# print(f"SUCCESS: Made it to the other side of experiment finished.", f"device:{torch.cuda.current_device()}")





# dotenv.load_dotenv(override=True)

# @hydra.main(config_path="configs/", config_name="multi-gpu")
@hydra.main(config_path="conf", config_name="base_conf")
def main(cfg: DictConfig):

	# Imports should be nested inside @hydra.main to optimize tab completion
	# Read more here: https://github.com/facebookresearch/hydra/issues/934
	# from lightning_hydra_classifiers.train_multi_gpu import train
	# from lightning_hydra_classifiers.utils import template_utils
	
	# A couple of optional utilities:
	# - disabling python warnings
	# - easier access to debug mode
	# - forcing debug friendly configuration
	# - forcing multi-gpu friendly configuration
	# You can safely get rid of this line if you don't want those
	# template_utils.extras(cfg)
	template_utils.initialize_config(cfg)
	# OmegaConf.set_struct(cfg, False)

	# Pretty print config using Rich library
	if cfg.get("print_config_only"):
		template_utils.print_config(cfg, resolve=True)
		return

	cfg = run_pretrain(cfg=cfg)
	
	return train(cfg)

# def initialize_config(cfg: DictConfig):
# 	OmegaConf.set_struct(cfg, False)
# 	OmegaConf.register_new_resolver("int", int)
# 	return cfg
		

# @hydra.main(config_path="conf", config_name="base_conf")
# def main(cfg: omegaconf.DictConfig):
#	 run(cfg)


if __name__ == "__main__":
	main()