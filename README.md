# CALIMOCHO

An implementation of Explanatory Active Learning (XAL) based on
Self-explainable Neural Networks.

See:

* Stefano Teso - *Toward Faithful Explanatory Interactive Machine Learning with Self-explainable Neural Nets*, Proceedings of the 3rd International Tutorial & Workshop on Interactive and Adaptive Learning (IAL'19).
* Stefano Teso and Kristian Kersting - *Explanatory Interactive Machine Learning*, International Conference on AI, Ethics and Society, 2019 ([pdf](https://ml-research.github.io/papers/teso2019aies_XIML.pdf)).


### Dataset

Our preliminary experiments use the synthetic colors dataset from:

* Andrew Ross, Michael C. Hughes, Finale Doshi-Velez - *Right for the right reasons: Training differentiable models by constraining their explanations*

The original data can be found on the [rrr repo](https://github.com/dtak/rrr).  We used the preprocessed dataset from the [caipi repo](https://github.com/stefanoteso/caipi).


### Experiments

To run CALIMOCHO, use the `main.py` script.  Type `python main.py --help` for
the list of options.

To run the experiments on the colors dataset:

* Download the `toy_colors.npz` file from the caipi repository linked above and place it into the `data` directory

* Execute `colors.sh` in the shell

The code will save all results in the `results` directory in pickle format, and plot them in PNG format too.


### Plots

To draw the final plots, unzip the zipped results files and run:
```bash
python draw.py lime-colors0-simplearch results-colors-lime/results/colors0__passive\=True__n\=None__k\=5__p\=0.2__T\=100__W\=101__P\=__e\=0.01__L\=0.9\,0.0__E\=1000__B\=None__s\=0__limer\=*.pickle -s q1 -m 10 11 12 13
python draw.py active-shallow-colors0-margin -s q2 results-colors-active-partialz/results/colors0__strategy\=margin__passive\=False__n\=None__k\=5__p\=0.0001__c\=*__T\=300__W\=101__P\=__e\=0.01__L\=0.1\,0.0__E\=100__B\=None__s\=0__trace.pickle results-colors-active-partialz/results/colors0__strategy\=margin__passive\=False__n\=None__k\=5__p\=0.0001__c\=1__T\=300__W\=101__P\=__e\=0.01__L\=0.0\,0.0__E\=100__B\=None__s\=0__trace.pickle
python draw.py active-deeper-colors0 -s q3 results-colors-active-partialz/results/colors0__strategy\=random__passive\=False__n\=None__k\=5__p\=0.0001__c\=1__T\=300__W\=*__P\=__e\=0.01__L\={0.0,0.1},*__E\=100__B\=None__s\=0__trace.pickle
```



### Requirements

* `python >= 3.5`
* `sklearn >= 0.21.0`
* `tensorflow >= 1.13.1`

Older versions may also work.


### Acknowledgements

This work has received funding from the European Research Council (ERC) under
the European Union’s Horizon 2020 research and innovation programme (grant
agreement No. [694980] SYNTH: Synthesising Inductive Data Models).
