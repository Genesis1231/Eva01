"""EVA01 — She's not an assistant. She's alive."""

import asyncio

from config import logger
from eva.core.app import wake


def main():
    logger.info("Starting EVA...")
    try:
        asyncio.run(wake())
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("EVA has gone to sleep.")


if __name__ == "__main__":
    main()
