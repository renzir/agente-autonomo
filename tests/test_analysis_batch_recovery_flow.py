
import asyncio
from unittest.mock import AsyncMock

from core.batcher import (
    AnalysisBatch,
    FileBlock,
    split_batch_for_retry,
)


def make_block(path: str, line_count: int = 10) -> FileBlock:
    content = "".join(
        f"line_{i}\n" for i in range(1, line_count + 1)
    )
    return FileBlock(
        path=path,
        lines_content=content,
        start_line=1,
        end_line=line_count,
    )


def make_batch(*blocks: FileBlock) -> AnalysisBatch:
    return AnalysisBatch(
        batch_id=1,
        files=list(blocks),
        total_lines=sum(block.line_count for block in blocks),
        total_chars=sum(block.char_count for block in blocks),
    )


def test_failed_batch_can_be_recovered_by_sub_batches():
    async def run_test():
        original = make_batch(
            make_block("a.py", 10),
            make_block("b.py", 15),
        )

        analyze = AsyncMock(
            side_effect=[
                RuntimeError("Error simulado de Ollama"),
                "Análisis correcto de a.py",
                "Análisis correcto de b.py",
            ]
        )

        try:
            await analyze(original)
            assert False, "Se esperaba un error simulado"
        except RuntimeError as exc:
            assert str(exc) == "Error simulado de Ollama"

        recovered = split_batch_for_retry(
            original, max_lines=300
        )

        assert len(recovered) == 2

        results = []
        for batch in recovered:
            results.append(await analyze(batch))

        assert results == [
            "Análisis correcto de a.py",
            "Análisis correcto de b.py",
        ]
        assert analyze.await_count == 3

    asyncio.run(run_test())

