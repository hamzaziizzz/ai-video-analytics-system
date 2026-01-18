from src.events.people_count import PeopleCountEngine


def test_people_count_triggers_after_stable_frames():
    engine = PeopleCountEngine(
        camera_id="cam1",
        min_count=1,
        stable_frames=3,
        cooldown_seconds=10.0,
        report_interval_seconds=60.0,
    )

    assert engine.update(0, 1000) is None
    assert engine.update(1, 2000) is None
    assert engine.update(1, 3000) is None

    event = engine.update(1, 4000)
    assert event is not None
    assert event.event_type == "people_count_alert"
    assert event.count == 1


def test_people_count_cooldown_suppresses_alerts():
    engine = PeopleCountEngine(
        camera_id="cam1",
        min_count=1,
        stable_frames=2,
        cooldown_seconds=5.0,
        report_interval_seconds=60.0,
    )

    engine.update(1, 1000)
    event = engine.update(1, 2000)
    assert event is not None

    engine.update(1, 3000)
    event = engine.update(1, 4000)
    assert event is None


def test_people_count_report_interval():
    engine = PeopleCountEngine(
        camera_id="cam1",
        min_count=10,
        stable_frames=3,
        cooldown_seconds=5.0,
        report_interval_seconds=2.0,
    )

    assert engine.update(0, 1000) is None
    report = engine.update(0, 4000)
    assert report is not None
    assert report.event_type == "people_count_report"
