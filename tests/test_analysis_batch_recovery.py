import pytest

from core.batcher import (
    AnalysisBatch,
    FileBlock,
    split_batch_for_retry,
    split_block,
)


def make_block(path: str, start_line: int, line_count: int) -> FileBlock:
    lines = [
        f"line_{n}\n"
        for n in range(start_line, start_line + line_count)
    ]
    return FileBlock(
        path=path,
        lines_content="".join(lines),
        start_line=start_line,
        end_line=start_line + line_count - 1,
    )


def make_batch(batch_id: int, blocks: list[FileBlock]) -> AnalysisBatch:
    return AnalysisBatch(
        batch_id=batch_id,
        files=blocks,
        total_lines=sum(block.line_count for block in blocks),
        total_chars=sum(block.char_count for block in blocks),
    )


def test_split_block_preserves_all_lines_and_ranges():
    block = make_block("demo.py", 1, 650)

    chunks = split_block(block, max_lines=300)

    assert len(chunks) == 3
    assert [
        (chunk.start_line, chunk.end_line)
        for chunk in chunks
    ] == [(1, 300), (301, 600), (601, 650)]

    assert "".join(chunk.lines_content for chunk in chunks) == (
        block.lines_content
    )
    assert sum(chunk.line_count for chunk in chunks) == block.line_count


def test_recovery_splits_multiple_files_into_individual_batches():
    first = make_block("a.py", 1, 10)
    second = make_block("b.py", 1, 15)
    original = make_batch(99, [first, second])

    recovered = split_batch_for_retry(original, max_lines=300)

    assert len(recovered) == 2
    assert [batch.files[0].path for batch in recovered] == [
        "a.py",
        "b.py",
    ]
    assert [batch.files[0].lines_content for batch in recovered] == [
        first.lines_content,
        second.lines_content,
    ]
    assert sum(batch.total_lines for batch in recovered) == (
        original.total_lines
    )
    assert sum(batch.total_chars for batch in recovered) == (
        original.total_chars
    )


def test_recovery_splits_large_file_without_losing_lines():
    original_block = make_block("large.py", 1, 650)
    original = make_batch(1, [original_block])

    recovered = split_batch_for_retry(original, max_lines=300)
    recovered_blocks = [batch.files[0] for batch in recovered]

    assert len(recovered_blocks) == 3
    assert [
        (block.start_line, block.end_line)
        for block in recovered_blocks
    ] == [(1, 300), (301, 600), (601, 650)]

    assert "".join(
        block.lines_content for block in recovered_blocks
    ) == original_block.lines_content

    assert sum(batch.total_lines for batch in recovered) == (
        original.total_lines
    )


def test_recovery_cannot_split_single_line_block():
    block = make_block("tiny.py", 1, 1)
    original = make_batch(1, [block])

    assert split_batch_for_retry(original, max_lines=300) == []


def test_recovery_returns_empty_for_empty_batch():
    original = AnalysisBatch(batch_id=1)

    assert split_batch_for_retry(original, max_lines=300) == []


@pytest.mark.parametrize("max_lines", [0, -1])
def test_split_block_rejects_invalid_max_lines(max_lines):
    block = make_block("demo.py", 1, 5)

    with pytest.raises(ValueError, match="max_lines"):
        split_block(block, max_lines=max_lines)