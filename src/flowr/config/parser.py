"""Put all dataparser configurations here."""
from typing import TYPE_CHECKING

import tyro
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig

from flowr.data.parser.dl3dv10k import DL3DV10KDataParserConfig
from flowr.data.parser.image import ImageDataParserConfig
from flowr.data.parser.scannetpp import ScanNetppDataParserConfig

all_dataparsers = {
    "image": ImageDataParserConfig(),
    "dl3dv": DL3DV10KDataParserConfig(),
    "scannetpp": ScanNetppDataParserConfig(),
}

if TYPE_CHECKING:
    DataParserUnion = DataParserConfig
else:
    DataParserUnion = tyro.extras.subcommand_type_from_defaults(
        all_dataparsers,
        prefix_names=False,
    )

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[DataParserUnion]
