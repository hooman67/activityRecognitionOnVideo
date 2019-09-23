import sys
import argparse
import pytest

sys.path.append('.')

def test_activity_model():
    """ Runs a network for two epochs with several steps_per_epoch. """
    from lstm_activity_model import main as activity_main
    args = argparse.Namespace()
    args.conf = "../config_test_activity.json"
    assert activity_main(args, is_debug=True)
