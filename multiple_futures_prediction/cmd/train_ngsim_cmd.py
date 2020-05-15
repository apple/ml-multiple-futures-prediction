from typing import List, Set, Dict, Tuple, Optional, Union, Any
from multiple_futures_prediction.train_ngsim import train, Params
import gin
import argparse

def parse_args() -> Any:
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='')
  return parser.parse_args()

def main() -> None:
  args = parse_args()
  gin.parse_config_file( args.config )
  params = Params()()
  train(params)

# python -m multiple_futures_prediction.cmd.train_ngsim_cmd --config multiple_futures_prediction/configs/mfp2_ngsim.gin
if __name__ == '__main__':
  main()


