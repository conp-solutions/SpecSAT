#!/usr/bin/python3

import argparse
import logging
import sys

# create logger
log = logging.getLogger(__name__)

VERSION = "0.0.1"


class CNFgenerator(object):
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('creating an instance of Auxiliary')


class SATsolver(object):
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)


class Benchmarker(object):
    def __init__(self, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.__dict__.update(kwargs)

    def run(self):
        self.log.debug("Starting Benchmarking Run")
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description='Run SpecSAT')
    parser.add_argument('-d', '--debug', default=False,
                        action='store_true', help='Log debug output')
    parser.add_argument('-v', '--version', default=False,
                        action='store_true', help='Print version of the tool')

    args = parser.parse_args()
    return vars(args)


def main():
    args = parse_args()

    if args.pop("debug"):
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S', level=logging.WARNING)

    if args.pop("version"):
        print("Version: {}".format(VERSION))
        return 0

    log.debug("Starting benchmarking with args: %r", args)
    benchmarker = Benchmarker(*args)
    return benchmarker.run()


if __name__ == "__main__":
    sys.exit(main())
