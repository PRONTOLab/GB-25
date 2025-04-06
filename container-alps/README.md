# Container for use on Alps @ CSCS

## Introduction

This directory contains the code to build a container for running the code on [Alps @ CSCS](https://www.cscs.ch/computers/alps) using the [Container Engine](https://eth-cscs.github.io/cscs-docs/software/container-engine/).
It builds the final SquashFS image in two steps:

* first a "base" container, which takes longer to build, which is published to [`ghcr.io/juliahpc/gb25-reactant-base:latest`](https://github.com/orgs/JuliaHPC/packages/container/package/gb25-reactant-base).
  This uses the most recent push to the `main` branch of this repository
* the "final" container, based on the previous one, which uses the local environment, to reflect local changes.

This setup is admittedly overcomplicated, could have easily used a single container, but we want to be flexible in case of necessary last minute changes, having a "base" image to build upon lets us save some time, probably not that much, estimated in the order of ~5 minutes, factoring in the cost of pushing the "base" image to the remote registry and pulling it again from there, but every minute counts when you have to race against time.

## Build and usage

To build the container on a compute node (which is [strongly recommended](https://confluence.cscs.ch/spaces/KB/pages/868834153/Building+container+images+on+Alps#BuildingcontainerimagesonAlps-BuildingimageswithPodman)), enter this directory and run the command

```sh
sbatch ./build.sh <image type>
```

where `<image type>` is one of

* `base`, for building the "base" image only and push it to the remote registry,
* `final`, for building the "final" image and make the SquashFS container and set it up for use in the Container Engine.
  If you use this option make sure you have a reasonably recent base image already pushed to the remote registry
* `all` to build both images in the same job.

You will need a [GitHub Personal Access Token](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic) with the ability to push a package to the [JuliaHPC](https://github.com/JuliaHPC/) organisation exported in an environment variable called `GHCR_TOKEN`.

If you want to submit the job to the debug queue, use

```sh
sbatch --partition=debug --time='00:30:00' ./build.sh <image type>
```

Once the "final" build is completed, [you can use it in a job](https://eth-cscs.github.io/cscs-docs/software/container-engine/#running-containerized-environments) with the command

```
srun --environment=reactant ...
```

## Acknowledgements

Thanks to Theofilos Manitaras (CSCS) for providing a first draft of the container setup.
All further complications are by Mosè Giordano (UCL).
