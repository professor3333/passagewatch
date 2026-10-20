#!/usr/bin/env bash
# The same-fish re-test's service on this machine (docs/review_retest.md): the review page
# and one worker running release passagewatch-0.3.3, with data in var/retest/ (not
# committed) on port 8012.
#
#   scripts/retest_service.sh start   # then send each reviewer their retest/T<n>.md
#   scripts/retest_service.sh status
#   scripts/retest_service.sh stop
set -euo pipefail

export REVIEW_SERVICE_DATA="var/retest"
export REVIEW_SERVICE_PORT="${PASSAGEWATCH_HTTP_PORT:-8012}"
export REVIEW_SERVICE_BUNDLE="bundles/passagewatch-0.3.3"
exec "$(dirname "$0")/review_sample_service.sh" "$@"
