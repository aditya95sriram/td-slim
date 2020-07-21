# SAT-based Local Improvement Method for Treedepth (TD-SLIM)



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3946663.svg)](https://doi.org/10.5281/zenodo.3946663)



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

> TD-SLIM(X) denotes the algorithm obtained by running TD-SLIM on top of 
> the heuristic solution provided by algorithm X

We consider two algorithms &ndash; TD-SLIM(DFS) and TD-SLIM(Sep), where
DFS is a randomized variant of the naive DFS heuristic 
by Villaamil [[1](#villaamil2017treedepth)] and Sep is one of the
separator-based heuristics from [[1](#villaamil2017treedepth)]
whose implementation was kindly provided 
by Tobias Oelschlägel [[2](#oelschlagel2014treewidth)].

We evaluated the performance on the 200 publicly available instances from the
PACE Challenge ([100 exact&downarrow;][4] and [100 heuristic&downarrow;][5]).
Out of the 200 instances, we filtered out 58 instances for which the 
heuristics were unable to compute a solution in 2 hours.
We compared TD-SLIM(DFS) and TD-SLIM(Sep) with the [best known values][6] from
the PACE Challenge, by running TD-SLIM for 30 minutes.
&Delta; denotes the depth computed by
TD-SLIM(X) minus the best known depth from the PACE Challenge.

|                | TD-SLIM(DFS) | TD-SLIM(Sep) |
|----------------|-------------:|-------------:|
| median &Delta; |            1 |            3 |
| &Delta;=0      |           53 |           41 |
| &Delta;&leq;1  |           73 |           58 |
| &Delta;&leq;5  |           95 |           88 |
| &Delta;&leq;10 |          107 |          101 |


Further, we evaluated the performance of running TD-SLIM on top of the 
[PACE Challenge 2020][2] winning heuristic - [ExTREEm][3].
We ran the heuristic for 15 minutes and then TD-SLIM on top of the heuristic 
provided solution for 15 minutes, making the total time limit 30 minutes, 
same as the PACE challenge.

| Instance  | ExTREEm | TD-SLIM | PACE best<sup>*</sup> |
|-----------|---------|---------|------------|
| exact_137 |      59 |      58 |         58 |
| heur_067  |     106 | **105** |        106 |
| heur_079  |     450 | **449** |        450 |
| heur_095  |      12 |      11 |         11 |
| heur_097  |     223 |     221 |        220 |
| heur_103  |     146 | **145** |        146 |

<sup>*</sup> - Best depth computed by any of the 55 heuristic solvers, 
or in the case of `exact_137`, the optimal treedepth


## References

1. <a name="villaamil2017treedepth">Villaamil, F. S. (2017). 
_About Treedepth and Related Notions_ 
(Doctoral dissertation, RWTH Aachen University).</a>
2. <a name="oelschlagel2014treewidth">Oelschlägel, T. (2014). 
_Treewidth from Treedepth_ 
(Bachelor's Thesis, RWTH Aachen University).</a>



[1]: https://pacechallenge.org/2020/verify.cpp
[2]: https://pacechallenge.org/2020/results/#heuristic-track
[3]: https://github.com/swacisko/pace-2020 
[4]: https://pacechallenge.org/files/pace2020-exact-public.tgz
[5]: https://pacechallenge.org/files/pace2020-heur-public.tgz
[6]: https://www.optil.io/optilion/problem/3177#tab-4
