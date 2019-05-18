import os

SLACK_INCOMMING_URL = os.getenv('SLACK_INCOMMING_URL', None)

DATASET_DIR = '/workdir/dataset'
INPUT_SHAPE = (3, 112, 112)

LFW_ROOT = os.path.join(DATASET_DIR, 'lfw-deepfunneled')
LFW_TEST_LIST = os.path.join(LFW_ROOT, 'lfw_test_pair.txt')