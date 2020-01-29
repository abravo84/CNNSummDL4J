#!/bin/bash

JAR_PATH="/home/upf/eclipse-workspace/cnnsumm/target/cnnsumm-0.0.1-SNAPSHOT-bin.jar"
CLASS_PATH="edu.upf.taln.cnnsumm.CnnSentencePredictionFinal"


CITATIONS_PATH="csv"

#TRAIN
TRAIN_PATH="/homedtic/abravo/corpora/SciSUM-2018/training/features/"
VEC_PATH=$TRAIN_PATH"googleacl_vec_sw_15.csv"
FEAT_PATH=$TRAIN_PATH"add_feat_vec_max_window.csv"

LABEL_PATH="/datasets/SciSUM-2018/training/features/"

NFEATS="29"
TRAIN_MODE="true"


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
ITERATIONS="1"
LEARNING_RATE="1e-1"
EPOCHS="3"
BATCH_SIZE="200"
ALGO_TYPE="alg2dual_4cnn"
CLASSES="1"
MODE="2d"
WINDOW="2"
NFILTERS="150"
NUMBER_CIT="5"


RESULT_PATH="/datasets/SciSUM-2018/results/"$FOLDERNAME



EXPERIMENT="sgar_human"





java -Xmx5G -XX:+UseSerialGC -cp $JAR_PATH $CLASS_PATH "$VEC_PATH" "$FEAT_PATH" "$LABEL_PATH" $HEIGHT $WIDTH $ITERATIONS $LEARNING_RATE $EPOCHS $BATCH_SIZE $NFILTERS $EXPERIMENT $ALGO_TYPE $CLASSES $MODE $WINDOW $TEST_MODE "$TEST_PATH" $TEST_VEC_PREFIX $TEST_FEAT_PREFIX "$RESULT_PATH" $CITATIONS_PATH $NUMBER_CIT $CITATIONS_PREFIX $NFEATS $TRAIN_MODE


