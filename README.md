[![Tests](https://github.com/benlipkin/optsent/actions/workflows/testing.yml/badge.svg)](https://github.com/benlipkin/optsent/actions/workflows/testing.yml)

# OptSent

### What is it?

A python module to generate optimal sequences of strings over built-in or user-defined transition objectives.

**Example use case**: Generate psycholinguistic stimulus materials that minimize compositionality over trial boundaries to avoid spillover effects.

### How can I get it?

Option 1: [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Make](https://www.gnu.org/software/make/)

```bash
git clone https://github.com/benlipkin/optsent
cd optsent
make env
```

Option 2: [Docker](https://docs.docker.com/get-docker/) (in progress)

```bash
docker pull benlipkin/optsent:latest
```

Option 3: [PyPi](https://pypi.org/) (in progress)

```bash
python -m pip install optsent
```

### How do I use it?

**CLI:**

```bash
usage: __main__.py [-h] [-o OUTDIR] [-m MODEL] [-j OBJECTIVE] [-s SOLVER] [-c CONSTRAINT] [-l SEQLEN] [-x] inputs

positional arguments (required):
inputs 						(path to CSV)

additional arguments (optional):
-h, --help  show this help message and exit
-o OUTDIR, --outdir OUTDIR 			(default: ./outputs/)
-m MODEL, --model MODEL				(default: gpt2 [can be any HuggingFace CausalLM])
-j OBJECTIVE, --objective OBJECTIVE		(default: logp(s1s2)-logp(s1)-logp(s2))
-s SOLVER, --solver SOLVER			(default: GreedyATSP)
-c CONSTRAINT, --constraint CONSTRAINT		(default: no word repeats on boundaries)
-l SEQLEN, --seqlen SEQLEN			(default: same length as input materials)
-x, --maximize					(default: false [minimize])

examples:
python -m optsent inputs/strings.csv
```

**API:** <sup>same options as CLI, but can also substitute user-defined objects (as long as they adhere to interface)</sup>

```python
from optsent import OptSent
from my_code import list_of_strings, custom_model, custom_objective

optsent = OptSent(
	list_of_strings,
	model=custom_model,
	objective=custom_objective,
	seqlen=100,
	maximize=True,
)

optsent.run()
```

**Sample Output**:

```bash
INFO:ArgTool:             inputs      inputs/SampleSmall.csv
INFO:ArgTool:             outdir      ~/optsent/outputs
INFO:ArgTool:             model       gpt2
INFO:ArgTool:             objective   normlogp
INFO:ArgTool:             solver      greedy
INFO:ArgTool:             constraint  repeats
INFO:ArgTool:             seqlen      -1
INFO:ArgTool:             maximize    False
INFO:ArgTool:             unique_id   min_SampleSmall_objective=normlogp_solver=greedy_constraint=repeats_model=gpt2
INFO:SentenceCollection:  Built collection of 10 sentences.
INFO:Model:               Loaded pretrained gpt2 model on cpu.
INFO:Objective:           Defined NormJointLogProb objective.
INFO:Optimizer:           Defined GreedyATSP solver.
INFO:OptSent:             Caching input strings.
INFO:OptSent:             Building transition graph.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:05<00:00, 18.84it/s]
INFO:OptSent:             Caching transition graph.
INFO:OptSent:             Solving sequence optimization.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 20132.66it/s]
INFO:OptSent:             Exporting optimal sequence.
INFO:CLI:                 Completed successfully in 0:00:08.883382.
```

### Where can I learn more?

Check out the [documentation](<>)!  (in progress)

### How can I contribute?

If you need a feature that is not currently included, or if you find a bug, please open an `issue` or `pull request` in this repo, and I will look into it for you.
