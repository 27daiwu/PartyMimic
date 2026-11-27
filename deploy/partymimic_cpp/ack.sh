#!/usr/bin/env bash
# V1.6.1: removes the e-stop latch file /tmp/partymimic_cpp_ESTOP, which exists ONLY if an operator created it with `./stop.sh --estop` (the service's own escalation latch is removed by the service itself).
# No prompt, no confirmation. The service never needs an acknowledgement any more: after every run it settles back to IDLE by itself.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
rm -f /tmp/partymimic_cpp_ESTOP; echo "e-stop latch cleared (the service itself needs no acknowledgement)."
