import os
import argparse
import numpy as np
import torch
from torch import nn
from preprocess import get_data
from models import CycleGAN, loss_function, accuracy_function, PositionalEncoding

#python assignment.py --type transformer --task both --data /Users/carolinezhang/Desktop/github-classroom/Brown-Deep-Learning/hw5-carolinebzhang/archive/data.p --chkpt_path /Users/carolinezhang/Desktop/github-classroom/Brown-Deep-Learning/hw5-carolinebzhang/code/testing_model
#you also need to pad the inputs
#python assignment.py --type rnn --task train --data /Users/carolinezhang/Desktop/github-classroom/Brown-Deep-Learning/hw5-carolinebzhang/archive/data.p
def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--genre',           required=True,              choices=['classical', 'jazz'],     help='Type of model to train')
    parser.add_argument('--task',           required=True,              choices=['train', 'test', 'both'],  help='Task to run')
    parser.add_argument('--data',           required=True,              help='File path to the assignment data file.')
    parser.add_argument('--testing',        type=str,                   help = "File path to a midi file to generate from")
    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--batch_size',     type=int,   default=100,    help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size',    type=int,   default=20,     help='Window size of text entries.')
    parser.add_argument('--chkpt_path',     default='',                 help='where the model checkpoint is')
    parser.add_argument('--check_valid',    default=True,               action="store_true",  help='if training, also print validation after each epoch')
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.


def main(args):

    ##############################################################################
    ## Data Loading
    training_data, training_labels = get_data(args.data)
    data_dict = {}
    feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 5, axis=0)
    img_prep  = lambda x: np.repeat(x, 5, axis=0)
    train_captions  = np.array(data_dict['train_captions'])
    test_captions   = np.array(data_dict['test_captions'])
    train_img_feats = feat_prep(data_dict['train_image_features'])
    test_img_feats  = feat_prep(data_dict['test_image_features'])
    # train_images    = img_prep(data_dict['train_images'])
    # test_images     = img_prep(data_dict['test_images'])
    word2idx        = data_dict['word2idx']
    # idx2word        = data_dict['idx2word']

    ##############################################################################
    ## Training Task
    if args.task in ('train', 'both'):
        
        ##############################################################################
        ## Model Construction

        model = CycleGAN()
        
        compile_model(model, args)
        train_model(
            model, train_captions, train_img_feats, word2idx['<pad>'], args, 
            valid = (test_captions, test_img_feats)
        )
        
        if args.chkpt_path: 
            ## Save model to run testing task afterwards
            
            save_model(model, args)
                
    ##############################################################################
    ## Testing Task
    if args.task in ('test', 'both'):
        if args.task != 'both': 
            ## Load model for testing. Note that architecture needs to be consistent
            model = load_model(args)
        if not (args.task == 'both' and args.check_valid):
            test_model(model, test_captions, test_img_feats, word2idx['<pad>'], args)

    ##############################################################################

##############################################################################
## UTILITY METHODS

def save_model(model, args):
    '''Loads model based on arguments'''
    
    torch.save(obj=model, f=args.chkpt_path)

    print(f"Model saved to '{args.chkpt_path}'")
    

def load_model(args):
    '''Loads model by reference based on arguments. Also returns said model'''
    model = nn.modules.load_model(
        args.chkpt_path,
        custom_objects=dict(
            PositionalEncoding = PositionalEncoding,
            CycleGANModel = CycleGAN
        ),
    )
    ## Saving is very nuanced. Might need to set the custom components correctly.
    ## Functools.partial is a function wrapper that auto-fills a selection of arguments. 
    ## so in other words, the first argument of ImageCaptionModel.test is model (for self)
    from functools import partial
    model.test    = partial(CycleGAN.test,    model)
    model.train   = partial(CycleGAN.train,   model)
    model.compile = partial(CycleGAN.compile, model)
    compile_model(model, args)
    print(f"Model loaded from '{args.chkpt_path}'")
    return model


def compile_model(model, args):
    '''Compiles model by reference based on arguments'''
    optimizer = torch.optim.Adam(lr=args.lr)
    model.compile(
        optimizer   = optimizer,
        loss        = loss_function,
        metrics     = [accuracy_function]
    )


def train_model(model, captions, img_feats, pad_idx, args, valid):
    '''Trains model and returns model statistics'''
    stats = []

    for epoch in range(args.epochs):
        try:
            for epoch in range(args.epochs):
                stats += [model.train(captions, img_feats, pad_idx, batch_size=args.batch_size)]
                print("stats")
                print(stats)
                if args.check_valid:
                    model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)
            return stats
        except KeyboardInterrupt as e:
            if epoch > 1:
                print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
            else: 
                raise e
        
        return stats

def test_model(model, captions, img_feats, pad_idx, args):
    '''Tests model and returns model statistics'''
    perplexity, accuracy = model.test(captions, img_feats, pad_idx, batch_size=args.batch_size)
    return perplexity, accuracy


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())