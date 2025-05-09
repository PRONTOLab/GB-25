#!/bin/bash

#SBATCH --job-name="build-container"
#SBATCH --time="1:00:00"
#SBATCH --output=build-container-%j.out
#SBATCH --error=build-container-%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=per_task:4
#SBATCH --constraint=gpu
#SBATCH --account=g191
#SBATCH --exclusive

set -euxo pipefail

if [[ "${@}" != "all" ]] &&  [[ "${@}" != "base" ]] && [[ "${@}" != "final" ]]; then
    echo "Must pass one of the arguments 'all', 'base' or 'final', got '${@}'"
    exit 1
fi

BASE_IMAGE="gb25-reactant-base"
BASE_TAG="latest"
BASE_REGISTRY="ghcr.io"
BASE_NAMESPACE="juliahpc"

if [[ "${@}" == "all" ]] || [[ "${@}" == "base" ]]; then

    # Here we build the base image, which takes more time and should be done
    # less frequently.

    # See
    # <https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic>
    # for pushing podman images to GHR. In particular you need a token with permissions:
    # * `read:packages` if you are only pulling images
    # * `write:packages` and `delete:packages` if you plan to push and delete

    echo "${GHCR_TOKEN}" | podman login ghcr.io -u giordano --password-stdin

    podman build -f Containerfile-base -t "${BASE_IMAGE}:${BASE_TAG}" .
    podman tag "${BASE_IMAGE}:${BASE_TAG}" "${BASE_REGISTRY}/${BASE_NAMESPACE}/${BASE_IMAGE}:${BASE_TAG}"
    podman push "${BASE_REGISTRY}/${BASE_NAMESPACE}/${BASE_IMAGE}:${BASE_TAG}"

    echo "Base image successfully built and pushed"

fi

if [[ "${@}" == "all" ]] || [[ "${@}" == "final" ]]; then

    IMAGE="reactant"
    TAG="latest"
    FILENAME="${IMAGE}.sqsh"
    TOML="${IMAGE}.toml"
    # Tricks like
    #     SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    # don't seem to work in slurm job scripts, so let's use something more dumb
    SCRIPT_DIR="$(realpath ${PWD})"
    export WORKDIR=$(dirname "${SCRIPT_DIR}")

    podman build --build-arg=DESTDIR="${WORKDIR}" --build-arg=BASE_IMAGE_TAG="${BASE_REGISTRY}/${BASE_NAMESPACE}/${BASE_IMAGE}:${BASE_TAG}" -f Containerfile -t "${IMAGE}:${TAG}" ..

    rm -fv "${FILENAME}"
    # For some reason the following command may exit with code 127 even if successful.
    # We make sure the output image is on disk with the `ls` below.
    enroot import -x mount -o "${FILENAME}" podman://"${IMAGE}:${TAG}" || true

    ls -lhrt "${FILENAME}"

    mkdir -vp "${HOME}/.edf"
    cat "${TOML}" | envsubst > "${HOME}/.edf/${TOML}"

    echo "Final image successfully built and set up"
fi
