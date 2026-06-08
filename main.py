"""EVA01, She is here."""

import asyncio

from config import logger
from eva.core.app import wake


def main():
    logger.info("And on the seventh day, you created Eva.")
    try:
        asyncio.run(wake())
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("EVA has gone back to sleep.")


if __name__ == "__main__":
    main()
