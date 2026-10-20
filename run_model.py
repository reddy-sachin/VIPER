"""Backwards-compatible entry point; same as the `viper` command.

    python run_model.py 2014-03-16              # download inputs and predict
    python run_model.py --input example_day.csv # use a prepared feature CSV
"""

from viper.cli import main

if __name__ == "__main__":
    main()
