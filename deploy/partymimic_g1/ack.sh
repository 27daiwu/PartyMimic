#!/usr/bin/env bash
# Acknowledge a HOLD (after a fault / unconfirmed exit / escalation). YOU state that the robot is safe and checked. Also clears the e-stop latch.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
echo "Only continue if the robot is physically safe (standing/supported/damped) and you read logs/runs/*/ and logs/service.log."; read -r -p "type YES to acknowledge: " a
[ "$a" = YES ] || { echo "not acknowledged"; exit 1; }
rm -f /tmp/partymimic_g1_ESTOP; touch logs/ACK; echo "ACK written (latch cleared). The service returns to IDLE within a second."
