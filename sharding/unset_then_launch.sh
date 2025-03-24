#!/bin/bash -l

# Important else XLA might hang indefinitely
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

julia --project --threads=auto -O0 ${HOME}/GB-25/sharding/sharded_baroclinic_instability.jl
