import argparse

import numpy as np
import torch
from torch import nn, optim
from torchtext import data, datasets

from nlp.model import SSTModel
from nlp.listops import ListOps


def evaluate(args):
    text_field = data.Field(lower=args.lower, include_lengths=True,
                            batch_first=True)
    label_field = data.Field(sequential=False)

    filter_pred = None
    if not args.fine_grained:
        filter_pred = lambda ex: ex.label != 'neutral'
    if args.task == 'sst':
        dataset_splits = datasets.SST.splits(
            root='data/', text_field=text_field, label_field=label_field,
            fine_grained=args.fine_grained, train_subtrees=True,
            filter_pred=filter_pred)
    elif args.task == 'listops':
        filter_pred = lambda ex: len(ex.text.split()) < 100
        dataset_splits = ListOps.splits(
            path='data/listops', text_field=text_field, label_field=label_field, filter_pred=filter_pred)
    else:
        print ('not supported')
        exit()
    test_dataset = dataset_splits[2]

    text_field.build_vocab(*dataset_splits)
    label_field.build_vocab(*dataset_splits)

    _, _, test_loader = data.BucketIterator.splits(
        datasets=dataset_splits, batch_size=args.batch_size, device=args.device)

    num_classes = len(label_field.vocab)
    model = SSTModel(num_classes=num_classes, num_words=len(text_field.vocab),
                     word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                     clf_hidden_dim=args.clf_hidden_dim,
                     clf_num_layers=args.clf_num_layers,
                     use_leaf_rnn=args.leaf_rnn,
                     bidirectional=args.bidirectional,
                     intra_attention=args.intra_attention,
                     use_batchnorm=args.batchnorm,
                     dropout_prob=args.dropout,
                     temperature=args.temperature,
                     mode=args.mode)
    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.word_embedding.weight.size())
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    model.to(args.device)
    torch.set_grad_enabled(False)
    num_correct = 0
    num_data = len(test_dataset)
    criterion = nn.CrossEntropyLoss()
    for batch in test_loader:
        words, length = batch.text
        label = batch.label
        logits = model(words=words, length=length)
        label_pred = logits.max(1)[1]
        loss = criterion(input=logits, target=label)
        num_correct_batch = torch.eq(label, label_pred).long().sum()
        num_correct_batch = num_correct_batch.item()
        num_correct += num_correct_batch

    print(f'Loss: {loss:.4f}')
    print(f'Accuracy: {num_correct / num_data:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--clf-hidden-dim', required=True, type=int)
    parser.add_argument('--clf-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--fine-grained', default=False, action='store_true')
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--mode', required=True, type=str)
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
