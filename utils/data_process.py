import collections

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


def my_tokenizer():
    """
    spacy 代表使用扩展模型
    使用预先训练的统计模型：de_core_news_sm 和 en_core_web_sm 分别对德语句子和英语句子进行分词
    :return: 分词器
    """
    tokenizer = {'de': get_tokenizer('spacy', language='de_core_news_sm'),
                 'en': get_tokenizer('spacy', language='en_core_web_sm')}
    return tokenizer


def build_vocab(tokenizer, filepath, min_freq, specials=None):
    """
    构建一个词典，使得 词 与 token 一一对应
    :param tokenizer: 分词器，由 my_tokenizer() 生成
    :param filepath: 用于构建词典的数据
    :param min_freq: 过滤掉小于 min_freq 的单词
    :param specials: 特殊字符。<unk>: 未知字符，<pad>: 用于padding的占位i而字符，<bos>: 句子起始字符， <eos>: 句子结束字符
                     <SEP>：两个句子之间的分隔符, <MASK>：填充被掩盖掉的字符
    :return: 词典
            print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
            # ['<unk>', '<pad>', '<bos>', '<eos>', '.', 'a', 'are', 'A', 'Two', 'in', 'men',...]
            print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；

            print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
            # {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3, '.': 4, 'a': 5, 'are': 6,...}
            print(vocab.stoi['are'])  # 通过单词返回得到词表中对应的索引
    """
    if specials is None:
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    counter = collections.Counter()
    with open(filepath, encoding='utf-8') as lines:
        for line in lines:
            counter.update(tokenizer(line))
    return Vocab(counter, specials=specials, min_freq=min_freq)


class LoadEnglishAndGermanDataset:
    def __init__(self, train_file_paths, tokenizer, batch_size=4, min_freq=3):
        """
        根据数据集建立各自的字典
        """
        self.tokenizer = tokenizer
        self.de_vocab = build_vocab(self.tokenizer['de'], filepath=train_file_paths[0], min_freq=min_freq)
        self.en_vocab = build_vocab(self.tokenizer['en'], filepath=train_file_paths[1], min_freq=min_freq)
        self.specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.PAD_IDX = self.de_vocab['<pad>']
        self.BOS_IDX = self.de_vocab['<bos>']
        self.EOS_IDX = self.de_vocab['<eos>']
        self.batch_size = batch_size

    def word2token(self, filepaths):
        """
        将各个数据集的词转换为 索引 形式，即 Token
        :param filepaths: 数据集的路径, 分别为德语和英语
        :return: 对应词表的 Token 形式
        """
        raw_de_iter = iter(open(filepaths[0], encoding='utf-8'))
        raw_en_iter = iter(open(filepaths[1], encoding='utf-8'))
        token_data = []
        for raw_de, raw_en in zip(raw_de_iter, raw_en_iter):
            # print([token for token in self.tokenizer['de'](raw_de.rstrip('\n'))])
            # # ['Eine', 'Gruppe', 'von', 'Männern', 'lädt', 'Baumwolle', 'auf', 'einen', 'Lastwagen']
            # print([self.de_vocab[token] for token in self.tokenizer['de'](raw_de.rstrip('\n'))])
            # # [15, 39, 25, 244, 2745, 0, 12, 21, 893]
            de_tensor = torch.tensor([self.de_vocab[token] for token in self.tokenizer['de'](raw_de.rstrip('\n'))],
                                     dtype=torch.long)
            en_tensor = torch.tensor([self.en_vocab[token] for token in self.tokenizer['en'](raw_en.rstrip('\n'))],
                                     dtype=torch.long)
            token_data.append([de_tensor, en_tensor])
        return token_data


if __name__ == '__main__':
    # # my_tokenizer() 测试
    # exam_tokenizer = my_tokenizer()
    # exam_sentence = "You're so so so beautiful!"
    # print(exam_tokenizer['en'](exam_sentence))
    # # ['You', "'re", 'so', 'so', 'so', 'beautiful', '!']

    # # build_vocab() 测试
    # exam_tokenizer = my_tokenizer()
    # data_path = '../data/val.en'
    # exam_vocab = build_vocab(exam_tokenizer['en'], data_path, 3)

    # LoadEnglishAndGermanDataset 测试
    train_file_paths = ['../data/train.de', '../data/train.en']
    exam_tokenizer = my_tokenizer()
    data_loader = LoadEnglishAndGermanDataset(train_file_paths, exam_tokenizer)
    val_file_paths = ['../data/val.de', '../data/val.en']
    data_loader.word2token(val_file_paths)



    pass
