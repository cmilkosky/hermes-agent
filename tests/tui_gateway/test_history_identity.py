"""Repeated content cannot stand in for persisted transcript identity."""
from tui_gateway.server import _dedupe_message_uids, _reconcile_display_with_live


def test_repeated_answer_keeps_later_unflushed_turn_by_row_identity():
    first = {'role': 'assistant', 'content': 'OK', '_row_id': 10}
    fresh = [{'role': 'user', 'content': 'again', '_row_id': 11},
             {'role': 'assistant', 'content': 'OK'}]
    assert _reconcile_display_with_live([first], [first, *fresh]) == [first, *fresh]
    other = dict(first, _row_id=99)
    assert _reconcile_display_with_live([first], [other]) == [first]


def test_legacy_projection_uses_ordered_prefix_not_last_equal_answer():
    first = [{'role': 'user', 'content': 'continue'}, {'role': 'assistant', 'content': 'OK'}]
    second = [{'role': 'user', 'content': 'continue'}, {'role': 'assistant', 'content': 'OK'}]
    assert _reconcile_display_with_live(first, first + second) == first + second


def test_live_only_row_before_the_boundary_keeps_the_unflushed_tail():
    # A row the DB never stores verbatim (injected note, expanded @file user text) must be
    # skipped, not used to exhaust the persisted prefix and drop the unflushed turn.
    persisted = [{'role': 'user', 'content': 'continue'}, {'role': 'assistant', 'content': 'OK'}]
    tail = [{'role': 'user', 'content': 'new'}, {'role': 'assistant', 'content': 'unflushed'}]
    injected = [persisted[0], {'role': 'system', 'content': 'note'}, persisted[1], *tail]
    assert _reconcile_display_with_live(persisted, injected) == persisted + tail
    expanded = [{'role': 'user', 'content': 'continue\n[expanded @file body]'}, persisted[1], *tail]
    assert _reconcile_display_with_live(persisted, expanded) == persisted + tail


def test_compacted_archive_copy_with_same_message_uid_renders_once():
    original = {"role": "assistant", "content": "answer", "message_uid": "msg-1", "_row_id": 10}
    archived = {"role": "assistant", "content": "answer", "message_uid": "msg-1", "_row_id": 99}

    assert _dedupe_message_uids([original, archived]) == [original]
    assert _reconcile_display_with_live([original, archived], [original]) == [original]


def test_equal_text_with_distinct_or_missing_uids_remains_distinct():
    messages = [
        {"role": "user", "content": "again", "message_uid": "msg-1"},
        {"role": "user", "content": "again", "message_uid": "msg-2"},
        {"role": "user", "content": "again"},
        {"role": "user", "content": "again"},
    ]

    assert _dedupe_message_uids(messages) == messages
