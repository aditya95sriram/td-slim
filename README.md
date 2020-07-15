# SAT-based Local Improvement Method for Treedepth (TD-SLIM)

Tool to improve heuristic treedepth decompositions by applying (Max)SAT-based
methods on local substructures.

## Setup

### Install Python requirements
```shell script
pip install -r requirements.txt
pip install -r optional_requirements.txt  # optional
```


### (Max)SAT solvers

Read `solvers/README.md` for instructions on setting up the (Max)SAT solvers.


### External Validity Checker

Download and compile the official validity checker provided by the PACE 
challenge from [here][1] and place the compiled binary in the project root.
The tool uses this checker at the end of its computation to confirm that the 
final (possibly improved) decomposition is valid.


### Temporary directory

The tool needs an existing directory to store the temporary encodings, 
this can be specified by means of the `--temp` switch.



## Usage

> Run `python local_improvement.py --help` to display information about 
> all the arguments and options


### Example

```shell script
python local_improvement.py -f path/to/graph.gr -o20 -m --heuristic dfs -en
```

The tool can be halted (say using `Ctrl+C`) at any point in time and it will
return the best solution computed so far as `sol1.tree`

> Behaviour might be unpredictable with disconnected input graphs.


### Heuristic solvers

A number of simple heuristics are supplied as a part of the tool,
these include:
* DFS variants (simple, multi-probe, two-step)
* Lex path
* Random

These heuristics can be invoked using the `--heuristic` switch.
> Note: In this case the total computation time would include the time 
> required to compute the heuristic solution. 

The easiest way to run TD-SLIM on any heuristic not mentioned above, is to run 
the heuristic separately and obtain the heuristic decomposition in the form of 
an edgelist file containing directed edges of the treedepth decomposition
(directed away from the root).
Ensure that the name of this file matches the input graph. For instance:
If the input graph is `my_graph.gr`, the file containing the decomposition
should be named `my_graph.edgelist`. 

- [ ] Todo: Accept starting decomposition in the standard `.tree` format 

Once the decomposition file is ready, invoke `local_improvement.py` with
the `--start-with` switch indicating the directory containing the decomposition. 


## Results

We evaluated the performance of running TD-SLIM on top of the 
[PACE Challenge 2020][2] winning heuristic - [ExTREEm][3].
We ran the heuristic for 15 minutes and then TD-SLIM on top of the heuristic 
provided solution for 15 minutes, making the total time limit 30 minutes, 
same as the PACE challenge.

| Instance  | ExTREEm | TD-SLIM | PACE best^ |
|-----------|---------|---------|------------|
| exact_137 |      59 |      58 |         58 |
| heur_067  |     106 | **105** |        106 |
| heur_079  |     450 | **449** |        450 |
| heur_095  |      12 |      11 |         11 |
| heur_097  |     223 |     221 |        220 |
| heur_103  |     146 | **145** |        146 |

^ - Best depth computed by any of the 55 heuristic solvers


[1]: https://pacechallenge.org/2020/verify.cpp
[2]: https://pacechallenge.org/2020/results/#heuristic-track
[3]: https://github.com/swacisko/pace-2020