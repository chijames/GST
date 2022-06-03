import os
import re
from torchtext import data

class Ex:
    text = None
    label = None
    

class ListOps(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        with open(path) as f:
            for line in f:
                label, line = line.strip().split('\t')
                line = line.replace('(', '').replace(')', '')
                line = re.sub(' +', ' ', line)
                line = line.replace('[', '[ ').strip()
                ex = Ex()
                setattr(ex, 'text', line)
                setattr(ex, 'label', label)
                examples.append(ex)
        
        super(ListOps, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, path='./interim',
               train='train.tsv', validation='valid.tsv', test='test.tsv', **kwargs):
        train_data = None if train is None else cls(
            os.path.join(path, train), text_field, label_field, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
