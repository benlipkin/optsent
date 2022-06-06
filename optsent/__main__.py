import argparse
import datetime

from optsent import OptSent
from optsent.abstract import Object


class CLI(Object):
    def __init__(self) -> None:
        super().__init__()
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument("inputs")
        self._parser.add_argument("-o", "--outdir", default=self._base / "outputs")
        self._parser.add_argument("-m", "--model", default="gpt2")
        self._parser.add_argument("-j", "--objective", default="normlogp")
        self._parser.add_argument("-z", "--optimizer", default="greedy")
        self._parser.add_argument("-c", "--constraint", default="none")
        self._parser.add_argument("-l", "--seqlen", default=-1)
        self._parser.add_argument("-x", "--maximize", action="store_true")
        self._parser.add_argument("-n", "--ncores", default=-1)

    def run_main(self) -> None:
        start = datetime.datetime.now()
        OptSent(**vars(self._parser.parse_args())).run()
        elapsed = datetime.datetime.now() - start
        self.info(f"Completed successfully in {elapsed}.")


if __name__ == "__main__":
    CLI().run_main()
