# CALIMOCHO

An implementation of Explanatory Active Learning (XAL) based on
Self-explainable Neural Networks.

See:

* Stefano Teso - *Toward Faithful Explanatory Interactive Machine Learning*, submitted to the 3rd International Tutorial & Workshop on Interactive and Adaptive Learning (IAL'19).
* Stefano Teso and Kristian Kersting - *Explanatory Interactive Machine Learning*, International Conference on AI, Ethics and Society, 2019.


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


### Requirements

* `python >= 3.5`
* `sklearn >= 0.21.0`
* `tensorflow >= 1.13.1`

Older versions may also work.


### Acknowledgements

This work has received funding from the European Research Council (ERC) under
the European Union’s Horizon 2020 research and innovation programme (grant
agreement No. [694980] SYNTH: Synthesising Inductive Data Models).
