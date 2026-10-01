# Third-party software

## FluidX3D

NeuroSim uses [FluidX3D](https://github.com/ProjectPhysX/FluidX3D) by Dr. Moritz Lehmann as its lattice Boltzmann solver. It is included as an unmodified git submodule in `external/FluidX3D`, pinned to a specific commit. FluidX3D is **not** written by the NeuroSim authors and is distributed under its own licence, `external/FluidX3D/LICENSE.md`, which applies to it and to any work that includes it.

Summary of the obligations (the licence text is authoritative):

- Use is permitted for public research, education or personal use. **Commercial use and military use are not permitted.**
- Altered versions must be plainly marked as altered. NeuroSim never edits the submodule; every change is a separate patch file whose name and contents identify it as a NeuroSim modification (`tools/**/patches/*.patch`).
- If binaries of altered versions, or data/results generated with them, are published, the altered source code must be published as well. This repository is public for that reason.
- AI models must not be trained on FluidX3D source code.
- Scientific publications arising from it must cite the articles listed at https://github.com/ProjectPhysX/FluidX3D#references.
- The licence notice must not be removed from any source distribution.

The name "FluidX3D" belongs to its author. NeuroSim is an independent project and is not affiliated with or endorsed by the FluidX3D author.
