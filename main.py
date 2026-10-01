"""
main.py — Punto de entrada CLI para el Agente Autónomo

Características:
- Utiliza ÚNICAMENTE core.agent.LocalAgent como punto de entrada.
- Mantiene un único ciclo de eventos asyncio para toda la sesión.
- Respeta la clasificación estricta de IntentClassifier (/chat vs /agente).
- Soporta respuestas normales y respuestas en streaming.
- Gestiona correctamente la limpieza de recursos asíncronos al salir.

Uso:
    python main.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from collections.abc import AsyncIterable

# Asegurar que el directorio raíz está en sys.path para imports
ROOT_DIR = str(Path(__file__).resolve().parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Configuración de logging básica
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('main')

# Importaciones directas del núcleo del proyecto
from core.agent import LocalAgent, create_agent
from core.state import Message


def print_help():
    """Muestra ayuda sobre los comandos disponibles."""
    print("\n[AYUDA] Comandos disponibles:")
    print("  /chat <mensaje>      -> Modo conversacional")
    print("  /agente <tarea>      -> Modo autónomo")
    print("  <texto sin prefijo>  -> Modo autónomo por defecto")
    print("  /exit, /quit         -> Salir del agente")
    print("  /help                -> Mostrar esta ayuda\n")


async def _print_response(response):
    if isinstance(response, AsyncIterable):
        print("\n", end="", flush=True)

        import time

        async for chunk, is_final in response:
            if chunk:
                print(
                    f"\n[{time.perf_counter():.3f}] CHUNK: {chunk!r}",
                    flush=True
                )

        print()
        return

    if isinstance(response, Message):
        if response.content:
            print(f"\n{response.content}")
        else:
            print("\n[INFO] El agente no devolvió contenido.")
        return

    if response is not None:
        print(f"\n{response}")
    else:
        print("\n[INFO] El agente no devolvió contenido.")

async def run_cli():
    """
    Bucle principal interactivo asíncrono.

    Crea el agente una vez y reutiliza sus clientes durante toda
    la sesión.
    """

    print("\n" + "=" * 60)
    print("  INICIANDO AGENTE AUTÓNOMO LOCAL")
    print("  Escribe '/chat hola' o '/agente crea un archivo'")
    print("  Escribe '/exit' para cerrar.")
    print("=" * 60 + "\n")

    # Inicialización ÚNICA del agente
    try:
        agent = create_agent()
        logger.info("Agente local inicializado correctamente.")

    except Exception as e:
        logger.critical(
            f"Error crítico al inicializar el agente: {e}"
        )

        print(
            "\n❌ No se pudo inicializar el agente. "
            "Verifica la configuración y las dependencias."
        )
        return

    try:
        while True:

            # Lectura de input sin bloquear el event loop
            loop = asyncio.get_running_loop()

            user_input = await loop.run_in_executor(
                None,
                lambda: input("\ntú> ").strip()
            )

            if not user_input:
                continue

            # Comandos de salida
            if user_input.lower() in (
                '/exit',
                '/quit',
                'exit',
                'quit'
            ):
                print(
                    "\nCerrando sesión... "
                    "Gracias por usar el agente."
                )
                break

            # Ayuda
            if user_input.lower() == '/help':
                print_help()
                continue

            # Procesamiento mediante LocalAgent
            print(
                "[TRACE] Enviando a LocalAgent.run()...",
                file=sys.stderr
            )

            try:
                response = await agent.run(user_input)

                # Mostrar respuesta normal o streaming
                await _print_response(response)

            except Exception as e:
                logger.error(
                    f"Error durante la ejecución del agente: {e}",
                    exc_info=True
                )

                print(
                    "\n❌ Ocurrió un error interno. "
                    "Revisa los logs.\n"
                )

    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario.")

    finally:
        await _cleanup_agent(agent)


async def _cleanup_agent(agent: LocalAgent):
    """Cierra los recursos asíncronos del agente."""

    if not agent:
        return

    try:
        # Cerrar cliente Ollama
        if (
            hasattr(agent, 'ollama_client')
            and hasattr(agent.ollama_client, 'close')
        ):
            await agent.ollama_client.close()
            logger.info("Cliente Ollama cerrado correctamente.")

        # Cerrar métricas
        if (
            hasattr(agent, 'metrics_logger')
            and hasattr(agent.metrics_logger, 'close')
        ):
            await agent.metrics_logger.close()

    except Exception as e:
        logger.warning(
            f"Advertencia al limpiar recursos: {e}"
        )


def main():
    """Entrada síncrona que inicia un único ciclo de eventos."""

    try:
        asyncio.run(run_cli())

    except RuntimeError as e:
        if (
            "asyncio.run() cannot be called "
            "from a running event loop" in str(e)
        ):
            logger.error(
                "Error crítico: Se intentó iniciar un nuevo "
                "event loop desde uno existente."
            )
        else:
            raise


if __name__ == "__main__":
    main()