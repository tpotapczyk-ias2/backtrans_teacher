"""Tokenizer with nltk/stanford.  """

# Standard Library
import argparse

# Third Party
from tqdm import tqdm
import nltk
import subprocess



def tokenize(inpath='', outpath='', delim=' ',lower=False):

    total = int(
        subprocess.check_output(['wc', '-l', inpath]).split()[0])

    with open(inpath, 'r') as ifh:
        with open(outpath, 'w') as ofh:

            for line in tqdm(ifh,total=total):
                tokenized = delim.join(nltk.word_tokenize(line))

                if lower:
                    tokenized  = tokenized.lower()
                ofh.write(tokenized + "\n")


if __name__ == '__main__':

    nltk.download('punkt')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inpath')
    parser.add_argument('-o', '--outpath')
    parser.add_argument('-l', '--lower',action='store_true')
    parser.add_argument('-d',
                        '--delim',
                        required=False,
                        default=" ",
                        help='delimiter, default=" "')

    opt = vars(parser.parse_args())

    tokenize(**opt)
