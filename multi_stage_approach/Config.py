from data_utils import shared_utils
from transformers import BertTokenizer, AutoTokenizer


class BaseConfig(object):
    def __init__(self, args):
        # common parameters
        self.epochs = args.epoch
        self.batch_size = args.batch
        self.device = args.device
        self.fold = args.fold

        # lstm model parameters setting
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers

        # common model mode setting
        self.model_mode = args.model_mode
        self.model_type = args.model_type
        self.dataset_name = args.dataset_name
        self.stage_model = args.stage_model
        self.program_mode = args.program_mode
        self.position_sys = args.position_sys

        self.premodel_path = args.premodel_path if args.premodel_path is not None else ""

        self.language = "vie"

        self.path = PathConfig(
            self.device, self.dataset_name, self.program_mode, self.premodel_path
        )
        self.val = GlobalConfig(self.position_sys)
        print('self.path.bert_model_path', self.path.bert_model_path)


class PathConfig(object):
    def __init__(self, device, dataset_name, program_mode, premodel_path):
        # store split train and test data file path
        dir_name = dataset_name
        dir_name = dir_name if program_mode in {"run", "test"} else "test_" + dir_name

        self.standard_path = {
            "train": "../data/{}/train.txt".format(dir_name),
            "dev": "../data/{}/dev.txt".format(dir_name),
            "test": "../data/{}/test.txt".format(dir_name)
        }

        self.vn_path = {
            "train": "../data/{}/train_0001.txt".format(dir_name)
        }

        self.annotated_path = {
            "train": "../data/{}/annotated_0001.txt".format(dir_name)
        }
        
        # nlp tool file path
        if device == "cpu":
            # self.stanford_path = r"D:/stanford-corenlp-full-2018-10-05"
            # self.bert_model_path = r"D:/base_uncased/" if dataset_name == "Camera-COQE" else r"D:/base_chinese/"
            self.stanford_path = "../deps/stanford/"#premodel_path + "stanford-corenlp-full-2018-02-27"
            # self.bert_model_path = "bert-base-uncased"#premodel_path + "base_uncased/" if dataset_name == "Camera-COQE" else premodel_path + "base_chinese/"
            self.vncorenlp_path = "../deps/vncorenlp"
            self.bert_model_path = "vinai/phobert-base-v2"
            self.GloVe_path = "."#premodel_path + "vector/glove.840B.300d.txt"
            self.Word2Vec_path = "../deps/lexvec.commoncrawl.300d.W.pos.vectors"#premodel_path + "vector/word2vec.txt"
        else:
            self.vncorenlp_path = "../deps/vncorenlp"
            self.stanford_path = "../deps/stanford/"#premodel_path + "stanford-corenlp-full-2018-02-27"
            # self.bert_model_path = "bert-base-uncased"#premodel_path + "base_uncased/" if dataset_name == "Camera-COQE" else premodel_path + "base_chinese/"
            self.bert_model_path = "vinai/phobert-base-v2"
            self.GloVe_path = "."#premodel_path + "vector/glove.840B.300d.txt"
            self.Word2Vec_path = "../deps/lexvec.commoncrawl.300d.W.pos.vectors"#premodel_path + "vector/word2vec.txt"

        self.pre_process_data = {
            "train": "../data/pre_process/{}_train_data.txt".format(dataset_name),
            "dev": "../data/pre_process/{}_dev_data.txt".format(dataset_name),
            "test": "../data/pre_process/{}_test_data.txt".format(dataset_name)
        }


class GlobalConfig(object):
    def __init__(self, position_sys):
        self.elems = ["subject", "object", "aspect", "predicate"]
        self.polarities = ["COM-", "EQL", "COM+", "None"]
        self.polarity_dict = {"COM-": -1,
                              "EQL": 0,
                              "COM+": 1,
                              "None": 0}

        if position_sys == "SPAN":
            self.position_sys = []
        else:
            self.position_sys = list(position_sys)

        self.special_id_map, self.norm_id_map = {"O": 0}, {"O": 0}
            # self.norm_id_map == {
            #  "O": 0,
            #  "B": 1,
            #  "M": 2,
            #  "E": 3,
            # }

            # special_id_map == {
            #  "O": 0,
            # "B-subject": 1,
            # "M-subject": 2,
            # "E-subject": 3,
            # "S-subject": 4,
            # "B-object": 5,
            # ...}

        self.norm_id_map = shared_utils.create_tag_mapping_ids([], self.position_sys, other_flag=True)
        self.special_id_map = shared_utils.create_tag_mapping_ids(self.polarities, self.position_sys, other_flag=True)

        self.invert_norm_id_map = {v: k for k, v in self.norm_id_map.items()}
        self.invert_special_id_map = {v: k for k, v in self.special_id_map.items()}
