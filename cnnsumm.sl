#!/bin/bash
#SBATCH -J scisumm
#SBATCH -p high
#SBATCH --ntasks=1 #-n number of cores
#SBATCH --mem-per-cpu=30G
#SBATCH --workdir=/homedtic/abravo
#SBATCH -o /homedtic/abravo/outputs/out_%N.%J.txt # STDOUT
#SBATCH -e /homedtic/abravo/outputs/err_%N.%J.txt # STDERR
#SBATCH -C intel #amd
#SBATCH --array=1-18:1

ml Java/1.8.0_121

JAR_PATH="/homedtic/abravo/software/dl4j-examples-0.9.1-bin_v1_csv.jar"
CLASS_PATH="org.deeplearning4j.examples.convolution.sentenceclassification.CnnSentencePredictionFinal"


#TRAIN
TRAIN_PATH="/homedtic/abravo/corpora/SciSUM-2018/training/features/"
VEC_PATH=$TRAIN_PATH"googleacl_vec_sw_15.csv"
FEAT_PATH=$TRAIN_PATH"add_feat_vec_max_window.csv"
CITATIONS_PATH="csv"
NFEATS="29"
TRAIN_MODE="false"



#TEST
TEST_MODE="true"
TEST_PATH="/homedtic/abravo/corpora/SciSUM-2018/testing/features"
TEST_PATH="/homedtic/abravo/corpora/SciSUM-2018/ahmed2019_2"
TEST_PATH="/homedtic/abravo/corpora/SciSUM-2018/sample"
TEST_PATH="/homedtic/abravo/corpora/ahmed/missing"
TEST_VEC_PREFIX="googleacl_vec_sw_15_"
TEST_FEAT_PREFIX="add_feat_vec_max_window_"
CITATIONS_PREFIX="add_feat_vec_max_window_"




FOLDERNAME="res_nov_missing"

HEIGHT="15"
WIDTH="300"
ITERATIONS="25"
LEARNING_RATE="1e-1"
EPOCHS="3"
BATCH_SIZE="200"
ALGO_TYPE="alg2dual_4cnn"
CLASSES="1"
MODE="2d"
WINDOW="2"
NFILTERS="150"
NUMBER_CIT="5"




RESULT_PATH="/homedtic/abravo/corpora/SciSUM-2018/results2019/"$FOLDERNAME

declare -a experiments
declare -a filenames

#########################
#    GOOGLE
#########################

#SUMMA
experiments[1]="summa_human"
experiments[2]="summa_abstract"
experiments[3]="summa_community"

labels[1]="labels_summa_human_vec_max.csv"
labels[2]="labels_summa_abstract_vec_max.csv"
labels[3]="labels_summa_community_vec_max.csv"

#ROUGE
experiments[4]="rouge_human"
experiments[5]="rouge_abstract"
experiments[6]="rouge_community"

labels[4]="labels_rouge2_human.csv"
labels[5]="labels_rouge2_abstract.csv"
labels[6]="labels_rouge2_community.csv"

#GOOGLE
experiments[7]="google_human"
experiments[8]="google_abstract"
experiments[9]="google_community"

labels[7]="labels_google_human_vec_max.csv"
labels[8]="labels_google_abstract_vec_max.csv"
labels[9]="labels_google_community_vec_max.csv"


#ACL
experiments[10]="acl_human"
experiments[11]="acl_abstract"
experiments[12]="acl_community"

labels[10]="labels_acl_human_vec_max.csv"
labels[11]="labels_acl_abstract_vec_max.csv"
labels[12]="labels_acl_community_vec_max.csv"

#AVG
experiments[13]="gar_human"
experiments[14]="gar_abstract"
experiments[15]="gar_community"

labels[13]="labels_gar_human.csv"
labels[14]="labels_gar_abstract.csv"
labels[15]="labels_gar_community.csv"

#AVG
experiments[16]="sgar_human"
experiments[17]="sgar_abstract"
experiments[18]="sgar_community"

labels[16]="labels_sgar_human.csv"
labels[17]="labels_sgar_abstract.csv"
labels[18]="labels_sgar_community.csv"



LABEL_PATH=$TRAIN_PATH${labels[${SLURM_ARRAY_TASK_ID}]}
EXPERIMENT=${experiments[${SLURM_ARRAY_TASK_ID}]}

echo $LABEL_PATH $EXPERIMENT

java -Xmx28G -XX:+UseSerialGC -cp $JAR_PATH $CLASS_PATH $VEC_PATH $FEAT_PATH $LABEL_PATH $HEIGHT $WIDTH $ITERATIONS $LEARNING_RATE $EPOCHS $BATCH_SIZE $NFILTERS $EXPERIMENT $ALGO_TYPE $CLASSES $MODE $WINDOW $TEST_MODE $TEST_PATH $TEST_VEC_PREFIX $TEST_FEAT_PREFIX $RESULT_PATH $CITATIONS_PATH $NUMBER_CIT $CITATIONS_PREFIX $NFEATS $TRAIN_MODE
