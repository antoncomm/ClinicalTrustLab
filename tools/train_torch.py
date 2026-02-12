import argparse
import datetime
from neurone.utils.runners import get_runner


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("config", help="path to the config file")
    parser.add_argument(
        "--engine",
        type=str,
        # choices=list(E2E.keys()),
        required=False,
        default="cpu",
    )

    # parser.add_argument("--fp16", action="store_true", default=False) #FIXME need to eradicate bug when using this

    parser.add_argument(
        "--name",
        type=str,
        help="name of the experiment",
        required=False,
        default="experiment_{}".format(datetime.datetime.now()),
    )

    parser.add_argument(
        "--tracker",
        type=str,
        help="name of the tracker",
        required=False,
        default="tensorboard",
    )

    parser.add_argument(
        "--cuda", type=str, help="numbers of used cudas", required=False, default=None
    )

    parser.add_argument("--seed", type=int, help="seed", required=False, default=42)

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="whether to overwrite the model or not",
        default=True,
    )

    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="whether to overwrite the model or not",
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to load and preprocess the data.",
        default=1,
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministic mode: slower but deterministic",
        default=False,
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    runner = get_runner(args)
    runner.run()


if __name__ == "__main__":
    main()
