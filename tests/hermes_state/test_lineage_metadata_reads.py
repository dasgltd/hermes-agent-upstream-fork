"""Lineage identity does not hydrate prompt bodies or repeatedly reread ancestors."""
from hermes_state import SessionDB


def test_long_lineage_uses_metadata_reads_with_a_linear_query_budget(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / 'state.db')
    chain = [f'conversation-{index}' for index in range(32)]
    try:
        for index, sid in enumerate(chain):
            db.create_session(sid, source='discord', parent_session_id=chain[index - 1] if index else None,
                              system_prompt='large prompt ' * 8000)
            if index < len(chain) - 1:
                db.end_session(sid, 'compression')
        queries, text_bytes = [], []
        read_one, read_all = db._read_one, db._read_all

        def record(rows):
            text_bytes.append(sum(len(value.encode()) for row in rows for value in row if isinstance(value, str)))

        def one(sql, params=()):
            queries.append(sql)
            row = read_one(sql, params)
            record([row] if row is not None else [])
            return row

        def all_rows(sql, params=()):
            queries.append(sql)
            rows = read_all(sql, params)
            record(rows)
            return rows

        monkeypatch.setattr(db, '_read_one', one)
        monkeypatch.setattr(db, '_read_all', all_rows)
        assert db.get_compression_lineage(chain[-1]) == chain
        assert sum(text_bytes) < 64_000, 'lineage lookup hydrated the conversation prompts'
        assert len(queries) <= 2 * len(chain), 'lineage repeatedly reread the same ancestors'
    finally:
        db.close()
