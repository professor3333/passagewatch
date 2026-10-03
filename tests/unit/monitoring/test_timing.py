from __future__ import annotations

import time

from passagewatch.monitoring.timing import StageTimes


def test_stages_accumulate_and_wrapped_iterators_charge_their_producer() -> None:
    times = StageTimes()

    def slow_items() -> object:
        for i in range(3):
            time.sleep(0.01)
            yield i

    with times.stage("work"):
        time.sleep(0.01)
    with times.stage("work"):
        pass
    consumed = []
    start = time.perf_counter()
    for item in times.wrap("produce", slow_items()):
        time.sleep(0.05)  # the consumer's time is not charged to "produce"
        consumed.append(item)
    loop = time.perf_counter() - start

    assert consumed == [0, 1, 2]
    assert times.seconds["work"] >= 0.01
    assert times.seconds["produce"] >= 0.03
    assert times.seconds["produce"] <= loop - 0.15 + 0.01  # minus the consumer's 3 x 0.05 s
    assert set(times.rounded()) == {"work", "produce"}
