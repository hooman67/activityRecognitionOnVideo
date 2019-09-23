import sys
import argparse
import pytest

sys.path.append('.')


""" Run with `pytest -s -v` in command line. `-s` option is to give std_out. 
Checks for simple runtime errors/exceptions, difficult to test for ML bugs.
Check for input to the network and output of the network for ML bugs. 
"""

def test_detection_main():
    """ Runs a network for two epochs with several steps_per_epoch. """
    from lstm_detection_model import main as detection_main
    args = argparse.Namespace()
    args.conf = "../config_test_detection.json"
    assert detection_main(args, is_debug=True)


# TODO: a test that overfits on small batch. It can test that loss decreases
# for several consecutive epochs.
