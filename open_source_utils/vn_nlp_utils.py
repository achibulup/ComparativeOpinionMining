import os
import copy
import py_vncorenlp
import logging

class vnCoreNlp(object):
    def __init__(self, sentences, config):
        """
        :param sentences: a list of sentence, [sentence1, sentence2......]
        """
        self.sentences = sentences
        self.config = config
        self.language = self.config.language
        if not os.path.exists(self.config.path.pre_process_data[self.language]):
            py_vncorenlp.download_model(save_dir=config.vncorenlp_path)
        self.nlp = py_vncorenlp.VnCoreNLP(save_dir=config.vncorenlp_path)
        

        # using set to store label type
        self.pos_dict, self.pos_index = {"PAD": 0}, 1
        self.dep_label_dict, self.dep_label_index = {}, 1
        self.vocab = {}

        # store the maximum length of sequence
        self.max_len = -1

        #read annotated sequences
        #each sequence includes a dict {'index', 'wordForm', 'posTag', 'nerLabel', 'head', 'depLabel} 
        #more info https://github.com/vncorenlp/VnCoreNLP/tree/master
        self.annotated = self.get_annotated_sentences()

    def get_annotated_sentences(self):
        annotated_list = []
        for sent in self.sentences:
            annotated_list.append(self.nlp.annotate_text(sent))
        return annotated_list

    def get_tokenizer(self) -> list[list[str]]:
        """
        :return: a list of token by VnCoreNLP tokenizer
        """
        input_tokens = []
        for aDict in self.annotated:
            token_list = []
            for word_list in aDict:
                token_list.append(word_list['wordForm'])
            input_tokens.append(token_list)

        return input_tokens

    # def get_pos_feature(self, pos_dict, pos_index):
    #     """
    #     :param: pos_dict:
    #     :param: pos_index:
    #     :return: a list of pos-tag, with id
    #     """
    #     self.pos_dict = copy.deepcopy(pos_dict)
    #     self.pos_index = pos_index
    #     pos_tags = {}
    #     for aDict in self.annotated:
    #         pos_list = {}
    #         for i in range(len(aDict)):
    #             word_list = aDict[i]
    #             pos_list.append(word_list['posTag'])
    #         pos_tags.append(pos_list)

    #     pos_feature = []
    #     for index in range(len(self.sentences)):

    #         # update pos-tag set
    #         for tag in pos_tags:
    #             if tag not in self.pos_dict:
    #                 self.pos_dict[tag] = self.pos_index
    #                 self.pos_index += 1

    #         pos_feature.append([self.pos_dict[tag] for tag in pos_tags])

    #     return pos_feature, self.pos_dict, self.pos_index

    # def get_dep_feature(self):
    #     """
    #     :return: dependency matrix and dependency label matrix
    #     """
    #     dep_matrix_feature, dep_label_feature = [], []

    #     dep_lists = {}
    #     for aDict in self.annotated:
    #         dep = {}
    #         for i in range(len(aDict)):
    #             word_list = aDict[i]
    #             dep.append(word_list['depLabel'])
    #         dep_lists.append(dep)

    #     for index in range(len(self.sentences)):
    #         dep_parse = self.nlp.dependency_parse(self.sentences[index])

    #         label_col = list(list(zip(*dep_parse))[0])
    #         out_node, in_node = list(list(zip(*dep_parse))[1]), list(list(zip(*dep_parse))[2])

    #         # define dep matrix and dep label matrix
    #         dep_matrix = [[0 for _ in range(len(out_node))] for j in range(len(out_node))]
    #         dep_label_matrix = copy.deepcopy(dep_matrix)

    #         # self loop
    #         for i in range(len(out_node)):
    #             dep_matrix[i][i] = 1

    #         # get dep_matrix and dep_label_matrix
    #         for i in range(len(out_node)):
    #             if out_node[i] == 0:
    #                 continue

    #             dep_matrix[out_node[i] - 1][in_node[i] - 1] = 1
    #             dep_matrix[in_node[i] - 1][out_node[i] - 1] = 1

    #             if label_col[i] not in self.dep_label_dict:
    #                 self.dep_label_dict[label_col[i]] = self.dep_label_index
    #                 self.dep_label_index = self.dep_label_index + 1

    #             dep_label_matrix[out_node[i] - 1][in_node[i] - 1] = self.dep_label_dict[label_col[i]]

    #         dep_matrix_feature.append(dep_matrix)
    #         dep_label_feature.append(dep_label_matrix)

    #     return dep_matrix_feature, dep_label_feature
