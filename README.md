# concept_library
This is a library for hierarchical concept composition and reasoning.

# Installation
Install the dependencies via
```code
pip install -r requirements.txt
```

# Structure
The concepts.py contains class definitions for concepts and operators. A concept is a 3-tuple of (concept name, concept probability model, concept graph).
* A concept name C is a string (e.g., "cat", "line") that denotes the concept:
* A concept probability model P(X, M, C) is a mapping that maps an observation X (e.g., an image containing cats and other distractors), a binary mask M, and a concept name C to a joint probability scalar. In this library, we use energy-based models (EBMs, the unnormalized log-likelihood of the probability) to represent the joint probability, which allows easy composition.
* A concept graph is a graph whose nodes are constituent concept names and edges are constituent relation names (e.g., the concept graph for "parallel-line" consists of two "line" concept name as nodes and a "parallel" relation name as edge). 

For more information about concepts, see the [slides](https://docs.google.com/presentation/d/1WAR4dZ0J2E-u3V_DgYBYTF4mDCmRk0FXI8GPlM2kqdQ/edit?usp=share_link) for [ZeroC](https://arxiv.org/abs/2206.15049) paper.

The models.py contains the class definitions of important network architectures for energy-based models and related architectures.

# Some projects using this library:
* [zeroc](https://github.com/snap-stanford/zeroc): Implementation for ["ZeroC: A Neuro-Symbolic Model for Zero-shot Concept Recognition and Acquisition at Inference Time"](https://arxiv.org/abs/2206.15049) (Wu et al., NeurIPS 2022)


# Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{wu2022zeroc,
title={Zeroc: A neuro-symbolic model for zero-shot concept recognition and acquisition at inference time},
author={Wu, Tailin and Tjandrasuwita, Megan and Wu, Zhengxuan and Yang, Xuelin and Liu, Kevin and Sosi{\v{c}}, Rok and Leskovec, Jure},
booktitle={Neural Information Processing Systems},
year={2022},
}
```
