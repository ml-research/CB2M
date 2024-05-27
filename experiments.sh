# ======================= Source code, datasets and preprocessing =======================
# Each python script generates. Setting -log_dir (and in some instances also -out_dir) can be used to customize the output folders.
# This file contains scripts both for training of the base CBM and for the CB2M experiments.
# The CB2M experiments were only done for the independent CBM training scheme, the joint training is kept for completeness

# The data generation generates 5 different folds for each dataset. The scripts here are exemplary for fold 0.



## ======================= CUB =======================
# The experiments on CUB are based on the paper "Concept Bottleneck Models" (Koh et al. 2020).
# To perform the experiments, download the official CUB dataset (CUB_200_2011), pretrained inception v3 network (pretrained)
# and the processed CUB data (CUB_processed) from their codalab sheet (https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2)
## ======================= Data Processing =======================
python3 CUB/data_processing.py -save_dir data_CUB/CUB_processed/class_attr_data_10 -data_dir data_CUB/CUB_200_2011 -filter_concepts

## ======================= CBM Training =======================
### Bottleneck Model
python3 experiments.py CUB Bottleneck -log_dir results/CUB/ConceptModel_fold_0/ -e 1000 -optimizer sgd -pretrained -use_aux -b 64 -weight_decay 0.00004 -lr 0.01 -bottleneck -data_dir data_CUB/CUB_processed/class_attr_data_10 -image_dir images -fold 0
### Independent Model
python3 experiments.py CUB Independent -log_dir results/CUB/IndependentModel_fold_0/ -e 500 -optimizer sgd -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -data_dir data_CUB/CUB_processed/class_attr_data_10 -image_dir images -fold 0
### Combined Inference
python3 inference.py CUB -model_dir results/CUB/ConceptModel_fold_0/best_model.pth -model_dir2 results/CUB/IndependentModel_fold_0/best_model.pth -eval_data test -bottleneck -use_sigmoid -log_dir results/CUB/IndependentModel -data_dir data_CUB/CUB_processed/class_attr_data_10 -image_dir images -fold 0

### Joint Model (not tested)
#### Concept loss weight = 0.01
python3 experiments.py CUB Joint -ckpt -log_dir results/CUB/Joint0.01Model_fold_0/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -end2end -data_dir data_CUB/CUB_processed/class_attr_data_10 -image_dir images -fold 0
python3 inference.py CUB -model_dir results/CUB/Joint0.01Model_fold_0/best_model.pth -eval_data test -log_dir results/CUB/Joint0.01Model -data_dir data_CUB/CUB_processed/class_attr_data_10 -image_dir images -fold 0
#### Concept loss weight = 0.01, with Sigmoid
python3 experiments.py CUB Joint -ckpt -log_dir results/CUB/Joint0.01SigmoidModel_fold_0/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -end2end -use_sigmoid -data_dir data_CUB/CUB_processed/class_attr_data_10 -image_dir images -fold 0
python3 inference.py CUB -model_dir results/CUB/Joint0.01SigmoidModel_fold_0/best_model.pth -eval_data test -log_dir results/CUB/Joint0.01SigmoidModel -data_dir data_CUB/CUB_processed/class_attr_data_10 -image_dir images -fold 0


## ======================= Preparations for CB2M =======================
python3 evaluation/cb2m_experiments.py precompute CUB -model_dir results/CUB/ConceptModel_fold_0/best_model.pth -model_dir2 results/CUB/IndependentModel_fold_0/best_model.pth -bottleneck -use_sigmoid -data_dir data_CUB/CUB_processed/class_attr_data_10 -image_dir images -fold 0
# -fold optional, otherwise generates results for all folds
python3 evaluation/ttie_experiments.py hyperparameter CUB -log_dir results/TTIE_Independent -data_dir data_CUB/CUB_processed/class_attr_data_10
## ======================= Detection Experiments =======================
# For all in this part: -fold optional, otherwise generates results for all folds
# Assumes models are stored in the here specified directories
python3 evaluation/ttie_experiments.py detection CUB -log_dir results/TTIE_Independent -data_dir data_CUB/CUB_processed/class_attr_data_10
python3 evaluation/ttie_experiments.py performance CUB -log_dir results/TTIE_Independent -data_dir data_CUB/CUB_processed/class_attr_data_10 -method ectp
python3 evaluation/ttie_experiments.py performance CUB -log_dir results/TTIE_Independent -data_dir data_CUB/CUB_processed/class_attr_data_10 -method ectp -baseline random
python3 evaluation/ttie_experiments.py performance CUB -log_dir results/TTIE_Independent -data_dir data_CUB/CUB_processed/class_attr_data_10 -method ectp -baseline softmax
## ======================= Generalization Experiments =======================
# for CUB: The augmented dataset variants are: jitter, blur, erase, salt, speckle
# use a different fraction of the interventional data by changing the data_frac parameter (e.g. to 0.5)
python3 evaluation/ttie_experiments.py generalization CUB -model_dir2 results/CUB/IndependentModel_fold_0/best_model.pth -log_dir results/TTIE_Independent -data_dir data_CUB/CUB_processed/class_attr_data_10 -test_aug jitter -fold 0 -data_frac 1.0



## ======================= Parity MNIST unbalanced =======================
## ======================= Data Processing =======================
python3 MNIST/data_processing.py MNIST -save_dir data_MNIST/MNIST_unbalanced -data_dir data_MNIST/MNIST -no_class 9 -include 250 -save_images

## ======================= CBM Training =======================
### Bottleneck Model
python3 experiments.py MNIST Bottleneck -log_dir results/MNIST_unbalanced/ConceptModel_fold_0/ -e 10 -optimizer sgd -pretrained -b 64 -weight_decay 0.00004 -lr 0.001 -bottleneck -data_dir data_MNIST/MNIST_unbalanced -image_dir data_MNIST/MNIST -fold 0
### Independent Model
python3 experiments.py MNIST Independent -log_dir results/MNIST_unbalanced/IndependentModel_fold_0/ -e 10 -optimizer sgd -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -data_dir data_MNIST/MNIST_unbalanced -image_dir data_MNIST/MNIST -fold 0
### Combined Inference
python3 inference.py MNIST -model_dir results/MNIST_unbalanced/ConceptModel_fold_0/best_model.pth -model_dir2 results/MNIST_unbalanced/IndependentModel_fold_0/best_model.pth -eval_data test -bottleneck -use_sigmoid -log_dir results/MNIST_unbalanced/IndependentModel -data_dir data_MNIST/MNIST_unbalanced -image_dir data_MNIST/MNIST -fold 0
### Finetuning the CBM on the validation set
python3 experiments.py MNIST Finetune -log_dir results/MNIST_unbalanced/ConceptModel_fold_0_finetuned/ -model_dir results/MNIST_unbalanced/ConceptModel_fold_0/best_model.pth -e 5 -optimizer sgd -b 64 -weight_decay 0.00004 -lr 0.001 -bottleneck -data_dir data_MNIST/MNIST_unbalanced/ -image_dir data_MNIST/MNIST -train_file val -val_file val -fold 0

## ======================= Preparations for CB2M =======================
python3 evaluation/cb2m_experiments.py precompute MNIST_unbalanced -model_dir results/MNIST_unbalanced/ConceptModel_fold_0/best_model.pth -model_dir2 results/MNIST_unbalanced/IndependentModel_fold_0/best_model.pth -bottleneck -use_sigmoid -data_dir data_MNIST/MNIST_unbalanced -image_dir data_MNIST/MNIST -fold 0
# -fold optional, otherwise generates results for all folds
python3 evaluation/cb2m_experiments.py hyperparameter MNIST_unbalanced -log_dir results/TTIE_Independent -data_dir data_MNIST/MNIST_unbalanced
## ======================= Detection Experiments =======================
# For all in this part: -fold optional, otherwise generates results for all folds
# Assumes models are stored in the here specified directories
python3 evaluation/ttie_experiments.py detection MNIST_unbalanced -log_dir results/TTIE_Independent -data_dir data_MNIST/MNIST_unbalanced
python3 evaluation/ttie_experiments.py performance MNIST_unbalanced -log_dir results/TTIE_Independent -method ectp -data_dir data_MNIST/MNIST_unbalanced
python3 evaluation/ttie_experiments.py performance MNIST_unbalanced -log_dir results/TTIE_Independent -method ectp -data_dir data_MNIST/MNIST_unbalanced -baseline random
python3 evaluation/ttie_experiments.py performance MNIST_unbalanced -log_dir results/TTIE_Independent -method ectp -data_dir data_MNIST/MNIST_unbalanced -baseline softmax
## ======================= Generalization Experiments =======================
python3 evaluation/ttie_experiments.py generalization MNIST_unbalanced -model_dir2 results/MNIST/IndependentModel_fold_0/best_model.pth -log_dir results/TTIE_Independent -fold 0 -data_dir data_MNIST/MNIST_unbalanced



## ======================= Generalization to SVHN =======================
## ======================= Data Processing =======================
# save images for the MNIST dataset does not require save_images if parity MNIST unbalanced has been done before
python3 MNIST/data_processing.py MNIST -save_dir data_MNIST/MNIST_unbalanced -data_dir data_MNIST/MNIST -save_images
python3 MNIST/data_processing.py SVHN -save_dir data_MNIST/SVHN_processed -data_dir data_MNIST/SVHN -save_images
## ======================= CBM Training =======================
### Bottleneck Model
python3 experiments.py MNIST Bottleneck -log_dir results/MNIST/ConceptModel_fold_0/ -e 10 -optimizer sgd -pretrained -b 64 -weight_decay 0.00004 -lr 0.001 -bottleneck -data_dir data_MNIST/MNIST -image_dir data_MNIST/MNIST -fold 0
### Independent Model
python3 experiments.py MNIST Independent -log_dir results/MNIST/IndependentModel_fold_0/ -e 10 -optimizer sgd -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -data_dir data_MNIST/MNIST -image_dir data_MNIST/MNIST -fold 0
### Combined Inference
python3 inference.py MNIST -model_dir results/MNIST/ConceptModel_fold_0/best_model.pth -model_dir2 results/MNIST/IndependentModel_fold_0/best_model.pth -eval_data test -bottleneck -use_sigmoid -log_dir results/MNIST/IndependentModel -data_dir data_MNIST/MNIST_unbalanced -image_dir data_MNIST/MNIST -fold 0
## ======================= Preparations for CB2M =======================
python3 evaluation/ttie_experiments.py precompute SVHN -model_dir results/MNIST/ConceptModel_fold_0/best_model.pth -model_dir2 results/MNIST/IndependentModel_fold_0/best_model.pth -bottleneck -use_sigmoid -data_dir data_MNIST/SVHN_processed -image_dir data_MNIST/SVHN -fold 0
# -fold optional, otherwise generates results for all folds
python3 evaluation/ttie_experiments.py hyperparameter SVHN -log_dir results/TTIE_Independent -data_dir data_MNIST/SVHN_processed
## ======================= Generalization Experiments =======================
python3 evaluation/ttie_experiments.py generalization SVHN -model_dir2 results/MNIST/IndependentModel_fold_0/best_model.pth -log_dir results/TTIE_Independent -fold 0 -data_dir data_MNIST/SVHN_processed



## ======================= Parity Color MNIST =======================
## ======================= Data Processing =======================
python3 CUB/data_processing.py -save_dir data_MNIST/CMNIST_processed -data_dir data_MNIST/CMNIST -save_dir_unconf data_MNIST/CMNIST_unconf -save_images

## ======================= CBM Training =======================
### Bottleneck Model
python3 experiments.py CMNIST Bottleneck -log_dir results/CMNIST/ConceptModel_fold_0/ -e 20 -optimizer sgd -pretrained -b 64 -weight_decay 0.00004 -lr 0.001 -bottleneck -data_dir data_MNIST/CMNIST_processed -image_dir data_MNIST/CMNIST -fold 0
### Independent Model
python3 experiments.py CMNIST Independent -log_dir results/CMNIST/IndependentModel_fold_0/ -e 20 -optimizer sgd -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -image_dir data_MNIST/CMNIST -data_dir data_MNIST/CMNIST_processed -fold 0
### Combined Inference
python3 inference.py CMNIST -model_dir results/CMNIST/ConceptModel_fold_0/best_model.pth -model_dir2 results/CMNIST/IndependentModel_fold_0/best_model.pth -eval_data test -bottleneck -use_sigmoid -log_dir results/CMNIST/IndependentModel_fold_0 -image_dir data_MNIST/CMNIST -data_dir data_MNIST/CMNIST_processed -fold 0

## ======================= Preparations for CB2M =======================
python3 evaluation/ttie_experiments.py precompute CMNIST -model_dir results/CMNIST/ConceptModel_fold_0/best_model.pth -model_dir2 results/CMNIST/IndependentModel_fold_0/best_model.pth -bottleneck -use_sigmoid -data_dir data_MNIST/CMNIST_unconf -image_dir data_MNIST/CMNIST -fold 0
# -fold optional, otherwise generates results for all folds
python3 evaluation/ttie_experiments.py hyperparameter CMNIST -log_dir results/TTIE_Independent -data_dir data_MNIST/CMNIST_unconf
## ======================= Detection Experiments =======================
# For all in this part: -fold optional, otherwise generates results for all folds
# Assumes models are stored in the here specified directories
python3 evaluation/ttie_experiments.py detection CMNIST -log_dir results/TTIE_Independent -data_dir data_MNIST/CMNIST_unconf
python3 evaluation/ttie_experiments.py performance CMNIST -log_dir results/TTIE_Independent -method ectp -data_dir data_MNIST/CMNIST_unconf
python3 evaluation/ttie_experiments.py performance CMNIST -log_dir results/TTIE_Independent -method ectp -data_dir data_MNIST/CMNIST_unconf -baseline random
python3 evaluation/ttie_experiments.py performance CMNIST -log_dir results/TTIE_Independent -method ectp -data_dir data_MNIST/CMNIST_unconf -baseline softmax
## ======================= Generalization Experiments =======================
python3 evaluation/ttie_experiments.py generalization CMNIST -model_dir2 results/CMNIST/IndependentModel_fold_0/best_model.pth -log_dir results/TTIE_Independent -fold 0 -data_dir data_MNIST/CMNIST_unconf





## ======================= CUB distribution shift =======================
## Original scripts from the CBM paper, using CUB with distribution shifts. Not used in the CB2M experiments right now
# they require the places365 dataset
### Generate Adversarial Data
python3 CUB/gen_cub_synthetic.py -cub_dir data_CUB/CUB_200_2011 -places_dir data_CUB/places365 -out_dir data_CUB/AdversarialData
### Generate pickle files from the adversarial data
python3 CUB/generate_new_data.py ChangeAdversarialDataDir -adv_data_dir data_CUB/AdversarialData/CUB_fixed -train_splits train val -out_dir data_CUB/CUB_processed/adversarial_fixed -data_dir data_CUB/CUB_processed/class_attr_data_10
python3 CUB/generate_new_data.py ChangeAdversarialDataDir -adv_data_dir data_CUB/AdversarialData/CUB_random -train_splits train val -out_dir data_CUB/CUB_processed/adversarial_random -data_dir data_CUB/CUB_processed/class_attr_data_10
python3 CUB/generate_new_data.py ChangeAdversarialDataDir -adv_data_dir data_CUB/AdversarialData/CUB_black -train_splits train val -out_dir data_CUB/CUB_processed/adversarial_black -data_dir data_CUB/CUB_processed/class_attr_data_10

### Concept Model
python3 experiments.py CUB Bottleneck -ckpt -log_dir results/CUB/ConceptAdversarialModel_fold_0/ -e 1000 -optimizer sgd -pretrained -use_aux -b 64 -weight_decay 0.00004 -lr 0.01 -bottleneck -image_dir data_CUB/AdversarialData/CUB_fixed/train -data_dir data_CUB/CUB_processed/adversarial_fixed
### Independent Model
python3 experiments.py CUB Independent -log_dir results/CUB/IndependentAdversarialModel/ -e 1000 -optimizer sgd -pretrained -use_aux -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -image_dir data_CUB/AdversarialData/CUB_fixed/train -use_sigmoid -data_dir data_CUB/CUB_processed/adversarial_fixed
### Combined Inference
python3 inference.py CUB -model_dir results/CUB/ConceptAdversarialModel_fold_0/best_model.pth -model_dir2 results/CUB/IndependentAdversarialModel/best_model.pth -eval_data test -bottleneck -image_dir data_CUB/AdversarialData/CUB_fixed/test/ -use_sigmoid -log_dir results/CUB/IndependentAdversarialModel -data_dir data_CUB/CUB_processed/adversarial_fixed

### Joint Model
python3 experiments.py cub Joint -ckpt -log_dir results/CUB/Joint0.01AdversarialModel_fold_0/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -end2end -image_dir data_CUB/AdversarialData/CUB_fixed/train -data_dir data_CUB/CUB_processed/adversarial_fixed
python3 inference.py CUB -model_dir results/CUB/Joint0.01AdversarialModel_fold_0/best_model.pth -eval_data test -image_dir data_CUB/AdversarialData/CUB_fixed/test/ -log_dir results/CUB/Joint0.01AdversarialModel -data_dir data_CUB/CUB_processed/adversarial_fixed

