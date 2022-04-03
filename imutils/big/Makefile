.PHONY: help

SEED=14
#TRAIN_SIZE=0.8
HERBARIUM_ROOT_DEFAULT="/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize"
CATALOG_DIR="${HERBARIUM_ROOT_DEFAULT}/catalogs"
SPLITS_DIR = ${CATALOG_DIR}/splits/train_size-${TRAIN_SIZE}

help:
	@echo "    all"
	@echo "        Prepare project to be used as a marvin package."
	@echo "    catalog"
	@echo "        1. Process raw json metadata into train + test csv catalog files."
	@echo "    splits"
	@echo "        2. Read catalog files and output train, val, and test csv splits."
	@echo "    shards"
	@echo "        3. Read catalog csv splits and serialize the full image dataset + key metadata to disk, using webdataset to produce several shards across tar archives."
	
catalog_files := ${OUT_DIR}/train_metadata.csv ${OUT_DIR}/test_metadata.csv

$(catalog_files): catalog
	echo "Making $(catalog_files) for catalog"
	make catalog
	#echo "Made catalog"
	
catalog:
	python ./make_herbarium_2022_catalog_df.py \
		--herbarium_source_dir ${HERBARIUM_ROOT_DEFAULT} \
		--target_dir ${CATALOG_DIR}


splits_files := $(SPLITS_DIR)/train.csv $(SPLITS_DIR)/val.csv $(SPLITS_DIR)/test.csv

$(splits_files): splits
	echo "Making $(splits_files) for splits"
	make splits
	#echo "Made splits"

splits: $(catalog_files)
	python ./make_train_val_splits.py \
		--source_dir ${CATALOG_DIR} \
		--splits_dir ${SPLITS_DIR} \
		--train_size ${TRAIN_SIZE} \
		--seed ${SEED}

shards: $(splits_files)
	echo "shard TBD"
	
	



