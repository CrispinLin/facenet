"generates own_pairs.txt for validation indexing"
import argparse
import sys
import os
import fnmatch


def main(args):
    with open(args.pairs, 'w') as outfile:
        outfile.write("10   300 \n")
        for i, f in enumerate(os.scandir(args.image_dir)):
            if not f.is_dir():
                continue
            if i > (args.number_positive - 1):
                break
            else:
                pnglist = [
                    i for i in os.listdir(f.path)
                    if fnmatch.fnmatch(i, '*.png')
                ]
                path0 = pnglist[0][:-4]
                path1 = pnglist[1][:-4]
                outfile.write(' '.join([f.name, path0, path1, '\n']))
        i = 0
        for f in os.scandir(args.image_dir):
            if not f.is_dir():
                continue
            if i > 2 * args.number_positive:
                break
            else:
                i = i + 1
                if i % 2 == 0:
                    pnglist = [
                        i for i in os.listdir(f.path)
                        if fnmatch.fnmatch(i, '*.png')
                    ]
                    path0 = pnglist[0][:-4]
                    outfile.write(f.name + ' ' + path0 + ' ')
                else:
                    pnglist = [
                        i for i in os.listdir(f.path)
                        if fnmatch.fnmatch(i, '*.png')
                    ]
                    path0 = pnglist[0][:-4]
                    outfile.write(' '.join([f.name, path0, '\n']))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'image_dir',
        type=str,
        help='Path to the data directory containing test images.')
    parser.add_argument(
        '--number_positive',
        type=int,
        help='Number of positive image pairs generated to test.',
        default=100)
    parser.add_argument(
        '--number_negative',
        type=int,
        help='Number of negative image pairs generated to test.',
        default=100)
    parser.add_argument(
        '--pairs',
        type=str,
        help='The file containing the pairs to use for validation.',
        default='data/own_pairs.txt')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
