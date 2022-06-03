import argparse
import logging
import os
import numpy as np
import random

from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torchtext import data, datasets

from nlp.model import SSTModel
from nlp.listops import ListOps


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

def seeding(seed=1234):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)    
    np.random.seed(seed)
    np.random.RandomState(seed)

    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #seed all gpus    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False


def train(args):
    seeding(args.seed)
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
    
    text_field.build_vocab(*dataset_splits, vectors=args.pretrained)
    label_field.build_vocab(*dataset_splits)

    logging.info(f'Initialize with pretrained vectors: {args.pretrained}')
    logging.info(f'Number of classes: {len(label_field.vocab)}')
    
    train_loader, valid_loader, _ = data.BucketIterator.splits(
        datasets=dataset_splits, batch_size=args.batch_size, device=args.device)
    
    model = SSTModel(num_classes=len(label_field.vocab), num_words=len(text_field.vocab),
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
    if args.pretrained:
        model.word_embedding.from_pretrained(text_field.vocab.vectors)
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    logging.info(f'Using device {args.device}')
    model.to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    optimizer = optimizer_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5,
        patience=args.validate_every * args.halve_lr_every, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'train'))
    valid_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'valid'))

    def run_iter(batch, is_training):
        model.train(is_training)
        words, length = batch.text
        label = batch.label
        logits = model(words=words, length=length)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        loss = criterion(input=logits, target=label)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=params, max_norm=5)
            optimizer.step()
        return loss, accuracy

    def add_scalar_summary(summary_writer, name, value, step):
        if torch.is_tensor(value):
            value = value.item()
        summary_writer.add_scalar(tag=name, scalar_value=value,
                                  global_step=step)

    num_train_batches = len(train_loader)
    validate_iter = num_train_batches // args.validate_every
    best_vaild_accuacy = 0
    iter_count = 0
    for epoch in range(args.max_epoch):
        for batch_iter, train_batch in enumerate(train_loader):
            train_loss, train_accuracy = run_iter(
                batch=train_batch, is_training=True)
            iter_count += 1
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='accuracy', value=train_accuracy, step=iter_count)
            if (batch_iter + 1) % validate_iter == 0:
                valid_loss_sum = valid_accuracy_sum = 0
                num_valid_batches = len(valid_loader)
                for valid_batch in valid_loader:
                    valid_loss, valid_accuracy = run_iter(
                        batch=valid_batch, is_training=False)
                    valid_loss_sum += valid_loss.item()
                    valid_accuracy_sum += valid_accuracy.item()
                valid_loss = valid_loss_sum / num_valid_batches
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='loss', value=valid_loss, step=iter_count)
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='accuracy', value=valid_accuracy, step=iter_count)
                scheduler.step(valid_accuracy)
                progress = train_loader.iterations / len(train_loader) + epoch
                logging.info(f'Epoch {progress:.2f}: '
                             f'valid loss = {valid_loss:.4f}, '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'mode-{args.mode}'
                                      f'-temp-{args.temperature}'
                                      f'-seed-{args.seed}'
                                      f'-epoch-{progress:.2f}'
                                      f'-vloss-{valid_loss:.4f}'
                                      f'-vacc-{valid_accuracy:.4f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    logging.info(f'Saved the new best model to {model_path}')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--clf-hidden-dim', required=True, type=int)
    parser.add_argument('--clf-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--l2reg', default=0.0, type=float)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--fix-word-embedding', default=False,
                        action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--max-epoch', required=True, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--omit-prob', default=0.0, type=float)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--fine-grained', default=False, action='store_true')
    parser.add_argument('--halve-lr-every', default=2, type=int)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--validate-every', default=2, type=int, help='run validation every 1 / validate-every epoch')
    parser.add_argument('--mode', required=True, type=str)
    parser.add_argument('--seed', default=1234, type=int)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
