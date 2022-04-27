from argparse import ArgumentParser
from datetime import datetime

from optsent import OptSent
from optsent.abstract import Object


class CLI(Object):
    def __init__(self) -> None:
        super().__init__()
        self._parser = ArgumentParser()
        self._parser.add_argument("inputs")
        self._parser.add_argument("-o", "--outdir", default=self._base / "outputs")
        self._parser.add_argument("-m", "--model", default="gpt2")
        self._parser.add_argument("-j", "--objective", default="normlogp")
        self._parser.add_argument("-s", "--solver", default="greedy")
        self._parser.add_argument("-c", "--constraint", default="repeats")
        self._parser.add_argument("-l", "--seqlen", default=-1)
        self._parser.add_argument("-x", "--maximize", action="store_true")

    def run_main(self) -> None:
        start = datetime.now()
        OptSent(**vars(self._parser.parse_args())).run()
        elapsed = datetime.now() - start
        self.log(f"Completed successfully in {elapsed}.")


if __name__ == "__main__":
    CLI().run_main()
