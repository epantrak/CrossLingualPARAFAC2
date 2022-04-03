from Code.Utils.timer import Timer
from Code.Utils.utils import *
import argparse
import pickle
import json
import glob
import sys
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    if FLAGS.target_dataset == 'mldoc':
        dataset_folder = os.path.join('..', 'Datasets', 'mldoc')
    else:
        raise ValueError('Select target_dataset mldoc')

    if FLAGS.masked:
        exp_folder = os.path.join('..', 'Exps', 'Wikimedia', 'masked_dataset', 'english-{}'.format(FLAGS.target_lang),
                                  'S{}-{}'.format(_N_SAMPLES_, FLAGS.tokenizer))
    else:
        exp_folder = os.path.join('..', 'Exps', 'Wikimedia', 'full_dataset', 'english-{}'.format(FLAGS.target_lang),
                                  'S{}-{}'.format(_N_SAMPLES_, FLAGS.tokenizer))

    run_folder = os.path.join(exp_folder, 'R{}'.format(FLAGS.R), 'run{}'.format(_RUN_))

    with open(os.path.join(run_folder, 'vocab', 'tknz_params.json'), 'r') as fp:
        tknz_params = json.load(fp)
    tknz_params['use_tfidf'] = True
    print(tknz_params)

    data_train, data_dev, data_test = [], [], []
    labels_train, labels_dev, labels_test = [], [], []

    for language in ['english', FLAGS.target_lang]:
        with open(os.path.join(run_folder, 'vocab', language + '_vocab.pickle'), 'rb') as handle:
            vocab = pickle.load(handle)

        if FLAGS.target_dataset == 'mldoc':
            df_train_class, df_dev_class, df_test_class = read_raw_data_mldoc(dataset_folder, _MLDOC_, language)
        else:
            df_train_class, df_dev_class, df_test_class = read_raw_data_amazon(dataset_folder, language,
                                                                               FLAGS.domain)

        _TIME_.start()
        _data_train, _data_dev, _data_test, _ = tokenize_data(language=language,
                                                              df_train=df_train_class,
                                                              df_dev=df_dev_class,
                                                              df_test=df_test_class,
                                                              vocab=vocab,
                                                              **tknz_params)
        _TIME_.stop(tag='Tokenize {}'.format(language), verbose=True)

        labels_train_temp, labels_dev_temp, labels_test_temp, classes = encode_labels(
            df_train_class, df_dev_class, df_test_class)

        data_train.append(_data_train.T)
        labels_train.append(labels_train_temp)
        data_test.append(_data_test.T)
        labels_test.append(labels_test_temp)
        data_dev.append(_data_dev.T)
        labels_dev.append(labels_dev_temp)

    best_test_acc, best_dev_acc = -1, -1
    models = glob.glob(os.path.join(exp_folder, 'R{}'.format(FLAGS.R), 'run{}'.format(_RUN_), 'models', '*.pkl'))
    for model in models:
        print('Loading model {}'.format(model))
        try:
            with open(model, 'rb') as file:
                par2_model = pickle.load(file)
        except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
            print("Not a valid file please try again")
            continue

        U = par2_model['U']
        train_embeddings = [U[id_language].T @ data_train[id_language] for id_language in range(2)]
        dev_embeddings = [U[id_language].T @ data_dev[id_language] for id_language in range(2)]
        test_embeddings = [U[id_language].T @ data_test[id_language] for id_language in range(2)]

        train_data, train_labels = train_embeddings[0].T, labels_train[0]
        dev_data, dev_labels = dev_embeddings[0].T, labels_dev[0]
        test_data, test_labels = test_embeddings[1].T, labels_test[1]

        accuracies = compute_class(X_tr=train_data,
                                   X_dev=dev_data,
                                   X_tst=test_data,
                                   lb_tr=train_labels,
                                   lb_dev=dev_labels,
                                   lb_tst=test_labels)

        if accuracies['test'] > best_test_acc:
            best_test_acc = accuracies['test']
            best_dev_acc = accuracies['dev']

    print('Best classification test accuracy {} and dev accuracy {}'.format(best_test_acc, best_dev_acc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--target_lang", default=None, type=str, required=True,
                        help="The name of the source language.")

    parser.add_argument("--target_dataset", default=None, type=str, required=True,
                        help="The name of the source language.")

    parser.add_argument("--tokenizer", default='nltk', type=str, required=True,
                        help="The type of the tokenizer.")

    parser.add_argument("--masked", type=str2bool, nargs='?', const=True, default=False,
                        help="Masked data or not.")

    parser.add_argument("--R", type=str2bool, nargs='?', const=True, default=False,
                        help="Rank of the model")

    FLAGS = parser.parse_args()

    if FLAGS.target_lang not in ['german', 'french', 'spanish', 'italian', 'japanese', 'russian', 'chinese']:
        raise ValueError('Target language should be one of de, fr, es, it, ja, ru".')

    if FLAGS.target_dataset not in ['mldoc', 'amazon']:
        raise ValueError('Target dataset should be one of "mldoc" or "amazon".')

    """
            ---Masked dataset max size---       ---Full dataset max size---
                    fr: 290440              |           fr: 692204
                    de: 35335               |           de: 80438
                    it: 100443              |           it: 254081
                    es: 449661              |           es: 1013931
                    ru: 109841              |           ru: 240890
                    zh: 37177               |           zh: 86915
                    ja: 33458               |           ja: 79204
    """

    if FLAGS.target_lang == 'french':
        # _N_SAMPLES_ = 290440 if FLAGS.masked else 692204
        _N_SAMPLES_ = 80000 if FLAGS.masked else 692204
    elif FLAGS.target_lang == 'german':
        _N_SAMPLES_ = 35335 if FLAGS.masked else 80438
    elif FLAGS.target_lang == 'italian':
        _N_SAMPLES_ = 100443 if FLAGS.masked else 254081
    elif FLAGS.target_lang == 'spanish':
        _N_SAMPLES_ = 449661 if FLAGS.masked else 1013931
    elif FLAGS.target_lang == 'russian':
        _N_SAMPLES_ = 109841 if FLAGS.masked else 240890
    elif FLAGS.target_lang == 'chinese':
        _N_SAMPLES_ = 37177 if FLAGS.masked else 86915
    elif FLAGS.target_lang == 'japanese':
        _N_SAMPLES_ = 33458 if FLAGS.masked else 79204
    else:
        raise ValueError('Select correct target language')

    _SAVE_EMBEDDINGS_ = False

    _SAVE_ = False

    _TIME_ = Timer()

    _RUN_ = 1

    _MLDOC_ = 1000  ### DO NOT CHANGE IT!!

    _CLASSIFIERS_ = ['LogReg', 'KNN', 'SVM']

    _CLASSIFIER_ = _CLASSIFIERS_[0]

    print(sys.argv[1:])
    main()
