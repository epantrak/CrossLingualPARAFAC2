import os
import sys

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '..')

from Code.Utils.utils import read_aligned_raw_data, tokenize_data
from distutils.dir_util import copy_tree
from Code.Parafac2 import Parafac2
from Code.Utils.timer import Timer
from pathlib import Path
import argparse
import shutil
import pickle
import json

from tensorboardX import SummaryWriter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_run_folder(exp_folder):
    run = 1
    run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    if not os.path.exists(run_folder):
        Path(run_folder).mkdir(parents=True, exist_ok=True)

        print("Path {} created".format(run_folder))
        return run_folder

    while os.path.exists(run_folder):
        run += 1
        run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    print("Path {} created".format(run_folder))
    return run_folder


def main():
    time = Timer()

    data_folder = os.path.join('..', 'Datasets', 'wikimedia', 'english-{}'.format(FLAGS.target_lang))

    if FLAGS.masked:
        save_exp_folder = os.path.join('..', 'Exps', 'Wikimedia', 'masked_dataset',
                                       'english-{}'.format(FLAGS.target_lang),
                                       'S{}-{}'.format(_N_SAMPLES_, FLAGS.tokenizer))
    else:
        save_exp_folder = os.path.join('..', 'Exps', 'Wikimedia', 'full_dataset',
                                       'english-{}'.format(FLAGS.target_lang),
                                       'S{}-{}'.format(_N_SAMPLES_, FLAGS.tokenizer))

    print(save_exp_folder)

    Path(save_exp_folder).mkdir(parents=True, exist_ok=True)
    tknz_params = {'cased': False,
                   'use_tfidf': False,
                   'norm': 'l2'}

    vocab_folder = os.path.join(save_exp_folder, 'vocab')
    Path(vocab_folder).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(vocab_folder, 'tknz_params.json'), 'w') as fp:
        json.dump(tknz_params, fp)

    data_train = []
    vectorizers = {}
    for language_code in ['english', FLAGS.target_lang]:
        time.start()

        # To tokenize data
        df_train = read_aligned_raw_data(data_folder, language_code, masked=FLAGS.masked)
        df_train = df_train.iloc[:_N_SAMPLES_]

        #  Tokenized sentences
        X_train_temp, _, _, vectorizer = tokenize_data(language=language_code,
                                                       df_train=df_train,
                                                       **tknz_params)

        data_train.append(X_train_temp.T)
        vectorizers.update({language_code: vectorizer})

        time.stop(tag='Tokenize {}'.format(language_code), verbose=True)

    time.start()
    # for run in range(1, _N_RUNS_ + 1, 1):
    # for FLAGS.R in range(2500, 3000, 500):
    print('Training model rank: {}...'.format(FLAGS.R))

    exp_folder = os.path.join(save_exp_folder, 'R{}'.format(FLAGS.R))

    Path(exp_folder).mkdir(parents=True, exist_ok=True)

    run_folder = check_run_folder(exp_folder)

    Path(vocab_folder).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(vocab_folder, 'english_vocab.pickle'), 'wb') as handle:
        pickle.dump(vectorizers['english'].vocabulary_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(vocab_folder, '{}_vocab.pickle'.format(FLAGS.target_lang)), 'wb') as handle:
        pickle.dump(vectorizers[FLAGS.target_lang].vocabulary_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree(vocab_folder, os.path.join(run_folder, 'vocab'))

    # Create log folder
    log_folder = os.path.join(run_folder, 'logs')
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    # Create model folder
    save_model_folder = os.path.join(run_folder, 'models')
    Path(save_model_folder).mkdir(parents=True, exist_ok=True)

    parameters = {"rank": FLAGS.R,
                  "max_m_iter": _N_EPOCHS_,
                  "error_tol": 1e-6,
                  'approx_fit_error': 10}

    # Save model parameters
    with open(os.path.join(run_folder, 'fit_parameters.json'), 'w') as fp:
        json.dump(parameters, fp, indent=4, sort_keys=True)

    tf_writer = SummaryWriter(log_folder)
    model = Parafac2(**parameters)
    for epoch in range(_N_EPOCHS_):
        mean_loss, loss = model.partial_fit(data_train)
        if tf_writer:
            tf_writer.add_scalars('Error', {'error': mean_loss}, epoch)
            for k in range(_N_LANGUAGES_):
                tf_writer.add_scalars('Error_X{}'.format(k), {'error': loss[k]}, epoch)

        epoch += 1
        if epoch % 1 == 0:
            print('m_iter {} - model error {}'.format(epoch, mean_loss))

        if epoch % 10 == 0:
            model_name = "model_iter" + str(epoch) + ".pkl"
            pkl_components = {"S": model.get_S(), "U": model.get_U()}
            with open(os.path.join(save_model_folder, model_name), 'wb') as file:
                pickle.dump(pkl_components, file, protocol=4)

    shutil.rmtree(vocab_folder)
    time.stop(tag='Training', verbose=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    _N_LANGUAGES_ = 2
    _N_EPOCHS_ = 25

    parser.add_argument("--target_lang", default=None, type=str, required=True,
                        help="The name of the target language.")

    parser.add_argument("--tokenizer", default='nltk', type=str, required=True,
                        help="The type of the tokenizer.")

    parser.add_argument("--masked", type=str2bool, nargs='?', const=True, default=False,
                        help="Masked data or not.")

    parser.add_argument("--R", type=str2bool, nargs='?', const=True, default=False,
                        help="Rank of the model")

    FLAGS = parser.parse_args()

    if FLAGS.target_lang not in ['german', 'french', 'spanish', 'italian', 'japanese', 'russian', 'chinese']:
        raise ValueError('Target language should be one of german, french, spanish, italian, japanese, russian".')

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

    print('### Target language: {} - Number of samples {} - tokenizer: {} - masked: {} ###'
          .format(FLAGS.target_lang, _N_SAMPLES_, FLAGS.tokenizer, FLAGS.masked))
    main()
