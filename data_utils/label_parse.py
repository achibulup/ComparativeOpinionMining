from data_utils import shared_utils
import json


class LabelParser(object):
    def __init__(self, labels_col, elems, intermittent=False):
        """
        :param labels_col: list of labels of all sentences
        :param elems: ["subject", "object", "aspect", "predicate", "label"]
        :param intermittent: True denote "predicate" using intermittent representation
        """
        self.labels_col = labels_col
        self.elems = elems
        self.intermittent = intermittent
        self.elem_index = {"subject": 0, "object": 1, "aspect": 2, "predicate": 3, "label": 4}

    def parse_sequence_label(self, split_symbol: str="&&", sent_col:list[str]|None=None) -> (list[dict], list[list[tuple]]):
        """
        :param split_symbol:
        :param sent_col:
        :param language
        :return: (
            list[dict {
              "subject": [(int, int)]
              "object": [(int, int)]
              "aspect": [(int, int)]
              "predicate": [(int, int, int)]
            }],
            list[[tuple, tuple, tuple, tuple, tuple]]
        )
        """
        NULL_LABEL = '{"subject":[],"object":[],"aspect":[],"predicate":[],"label":""}'
        pair_tuple_col, elem_representation_col = [], []


        for row_index in range(len(self.labels_col)):
            sentence = sent_col[row_index]

            if self.labels_col[row_index][0] == NULL_LABEL:
                # print("?")
                pair_tuple_col.append([[(-1, -1)] * 5])
                print(pair_tuple_col)
                elem_representation_col.append(self.init_elem_representation())
                print(elem_representation_col)
                continue

            elem_set = self.init_elem_representation() # dict {"subject": [tuple], "object": [tuple], "aspect": [tuple], "predicate": [tuple]}

            pair_tuple_sequence = []
            for label in self.labels_col[row_index]:
                # print('label: ', label)
                elem_set, cur_pair_tuple = self.parse_each_pair_label(
                    label, elem_set, split_symbol, sentence, language
                )
                pair_tuple_sequence.append(cur_pair_tuple)


            # print('pair_tuple_sequence: ', pair_tuple_sequence)
            # print('elem_set: ', elem_set)
            pair_tuple_col.append(pair_tuple_sequence)
            elem_representation_col.append(elem_set)

        return elem_representation_col, pair_tuple_col

    def parse_each_pair_label(self, label, elem_set, split_symbol, sent=None, language="cn"):
        """
        :param label:
        :param elem_set:
        :param split_symbol:
        :param sent:
        :param language:
        :return: (
            dict {
              "subject": (int, int)
              "object": (int, int)
              "aspect": (int, int)
              "predicate": (int, int, int)
            },
            [tuple, tuple, tuple, tuple, tuple]
        )
        """
        parsed_label = json.loads(label)
        # elem_representation = shared_utils.split_string(label[1:-1], ";")
        pair_tuple_representation = [None] * 5
        result_elem = [-1, -1, 0]
        for key, value in parsed_label.items():
            elem_tuple = (-1, -1)
            if key != "label":
                if len(value) > 0:
                    if language == "cn":
                        s_index = int(shared_utils.split_string(value[0], split_symbol)[0])
                        e_index = int(shared_utils.split_string(value[-1], split_symbol)[0]) + 1
                    else:
                        s_index = int(shared_utils.split_string(value[0], split_symbol)[0]) - 1
                        e_index = int(shared_utils.split_string(value[-1], split_symbol)[0])

                    elem_tuple = (s_index, e_index)

            else:
                polarity = 1 if value[-1] == '+' else -1 if value[-1] == '-' else 0
                elem_tuple = (polarity, polarity)
                result_elem[2] = polarity
            
            if key == "predicate":
                result_elem[0:2] = elem_tuple
            elif key != "label":
                elem_set[key].add(elem_tuple)

            pair_tuple_representation[self.elem_index[key]] = elem_tuple

        elem_set["predicate"].add(tuple(result_elem))

        return elem_set, pair_tuple_representation

    @staticmethod
    def get_sub_elem(number_char_col, split_symbol):
        """
        :param number_char_col:
        :param split_symbol:
        :return:
        """
        elem_str = ""
        for num_char in number_char_col:
            elem_str += shared_utils.split_string(num_char, split_symbol)[1]

        return elem_str

    def init_elem_representation(self):
        return {key: set() for key in self.elems}
