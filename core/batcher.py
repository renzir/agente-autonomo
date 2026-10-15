
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class FileBlock:
    """Fragmento de un archivo, con su ruta y rango original de líneas."""
    path: str
    lines_content: str
    start_line: int
    end_line: int

    @property
    def line_count(self) -> int:
        if self.start_line > self.end_line:
            return 0
        return self.end_line - self.start_line + 1

    @property
    def char_count(self) -> int:
        return len(self.lines_content)


@dataclass
class AnalysisBatch:
    """Archivos o bloques enviados juntos a una llamada de análisis."""
    batch_id: int
    files: List[FileBlock] = field(default_factory=list)
    total_lines: int = 0
    total_chars: int = 0

    def add_block(self, block: FileBlock) -> None:
        self.files.append(block)
        self.total_lines += block.line_count
        self.total_chars += block.char_count


def _read_full_file(
    workspace_root: Path,
    relative_path: str,
) -> FileBlock:
    """Lee un archivo completo y conserva su ruta relativa al workspace."""
    root = workspace_root.resolve()
    candidate = (root / relative_path).resolve()

    # Impedir rutas fuera del workspace.
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Ruta fuera del workspace: {relative_path}"
        ) from exc

    if not candidate.is_file():
        raise FileNotFoundError(
            f"No existe el archivo: {relative_path}"
        )

    with candidate.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    return FileBlock(
        path=candidate.relative_to(root).as_posix(),
        lines_content="".join(lines),
        start_line=1,
        end_line=len(lines),
    )


def split_block(
    block: FileBlock,
    max_lines: int = 300,
) -> List[FileBlock]:
    """Divide un bloque conservando los números de línea originales."""
    if max_lines < 1:
        raise ValueError("max_lines debe ser mayor que cero")

    if block.line_count == 0:
        return [block]

    lines = block.lines_content.splitlines(keepends=True)
    chunks = []

    for offset in range(0, len(lines), max_lines):
        chunk = lines[offset:offset + max_lines]

        chunks.append(
            FileBlock(
                path=block.path,
                lines_content="".join(chunk),
                start_line=block.start_line + offset,
                end_line=block.start_line + offset + len(chunk) - 1,
            )
        )

    return chunks


def split_file(
    workspace_root: Path,
    relative_path: str,
    max_lines: int = 300,
) -> List[FileBlock]:
    """Lee un archivo y lo divide. Se reserva para recuperación por fallos."""
    full_block = _read_full_file(workspace_root, relative_path)
    return split_block(full_block, max_lines)


def create_batches(
    workspace_root: Path,
    relative_paths: List[str],
    max_lines_per_batch: int = 300,
    max_chars_per_batch: int = 40000,
) -> List[AnalysisBatch]:
    """
    Prepara los batches iniciales.

    - Un único archivo: se intenta enviar completo.
    - Varios archivos pequeños: se agrupan respetando ambos límites.
    - Archivos grandes dentro de una selección múltiple: cada uno va aparte,
      completo inicialmente.
    - La división adaptativa tras fallos se realiza después, desde el
      Orchestrator; este método no llama a Ollama.
    """
    if max_lines_per_batch < 1:
        raise ValueError("max_lines_per_batch debe ser mayor que cero")

    if max_chars_per_batch < 1:
        raise ValueError("max_chars_per_batch debe ser mayor que cero")

    if not relative_paths:
        return []

    # Evitar analizar dos veces la misma ruta.
    unique_paths = list(dict.fromkeys(relative_paths))

    # Leer y validar los archivos una vez al preparar los batches.
    full_blocks = [
        _read_full_file(workspace_root, path)
        for path in unique_paths
    ]

    # Caso especial: un único archivo se intenta analizar completo.
    if len(full_blocks) == 1:
        batch = AnalysisBatch(batch_id=1)
        batch.add_block(full_blocks[0])
        return [batch]

    batches: List[AnalysisBatch] = []
    current = AnalysisBatch(batch_id=1)

    def flush_current() -> None:
        nonlocal current

        if current.files:
            batches.append(current)
            current = AnalysisBatch(
                batch_id=len(batches) + 1
            )

    for block in full_blocks:
        is_large = (
            block.line_count > max_lines_per_batch
            or block.char_count > max_chars_per_batch
        )

        # Los archivos grandes nunca se agrupan con otros.
        if is_large:
            flush_current()

            batch = AnalysisBatch(batch_id=len(batches) + 1)
            batch.add_block(block)
            batches.append(batch)

            current = AnalysisBatch(batch_id=len(batches) + 1)
            continue

        would_exceed_lines = (
            current.total_lines + block.line_count
            > max_lines_per_batch
        )
        would_exceed_chars = (
            current.total_chars + block.char_count
            > max_chars_per_batch
        )

        if current.files and (
            would_exceed_lines or would_exceed_chars
        ):
            flush_current()

        current.add_block(block)

    flush_current()
    return batches


def split_batch_for_retry(
    batch: AnalysisBatch,
    max_lines: int = 300,
) -> List[AnalysisBatch]:
    """
    Prepara una división para recuperación después de reintentos fallidos.

    - Si hay varios archivos, los separa primero.
    - Si solo hay un archivo, lo divide por líneas.
    - Devuelve una lista vacía si no puede subdividirlo.
    """
    if max_lines < 1:
        raise ValueError("max_lines debe ser mayor que cero")

    if len(batch.files) > 1:
        return [
            AnalysisBatch(
                batch_id=index,
                files=[block],
                total_lines=block.line_count,
                total_chars=block.char_count,
            )
            for index, block in enumerate(batch.files, start=1)
        ]

    if not batch.files:
        return []

    block = batch.files[0]

    if block.line_count < 2:
        return []

    # Si ya cabe en el límite, dividir por la mitad para intentar recuperar.
    chunk_size = (
        max_lines
        if block.line_count > max_lines
        else (block.line_count + 1) // 2
    )

    subblocks = split_block(block, chunk_size)

    if len(subblocks) < 2:
        return []

    return [
        AnalysisBatch(
            batch_id=index,
            files=[subblock],
            total_lines=subblock.line_count,
            total_chars=subblock.char_count,
        )
        for index, subblock in enumerate(subblocks, start=1)
    ]