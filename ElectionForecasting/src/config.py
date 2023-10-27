from enum import Enum, auto
import datetime
import logging

from ElectionForecasting.src.utils.general import (
    configure_logging, create_directory
)

# Configure Logging
configure_logging()
DATE_TODAY = datetime.datetime.today().replace(
    tzinfo=datetime.timezone.utc
).strftime('%Y-%m-%d')


logging.info(f'SETTING DATE TO {DATE_TODAY}')

class Parties(Enum):
    CC = auto()
    DGM = auto()
    PDAL = auto()
    SSP = auto()

class Provinces(Enum):
    Cerebrica  = 'Cerebrica'
    Cortexia  = 'Cortexia'
    Neuronia  = 'Neuronia'
    Amperville  = 'Amperville'
    Binaryshire  = 'Binaryshire'
    ByteforgeDomain = 'Byteforge Domain'
    Circuiton  = 'Circuiton'
    Electropolis  = 'Electropolis'
    InfinitronPeninsula = 'Infinitron Peninsula'
    Infoglen  = 'Infoglen'
    Quantumridge  = 'Quantumridge'
    Voltage  = 'Voltagea'

restricted_party = Parties.SSP.name.lower()
unrestricted_provinces = [Provinces.Cerebrica.name, Provinces.Cortexia.name, Provinces.Neuronia.name]
party_order = [party.name.lower() for party in Parties]
province_order = [province.value for province in Provinces]
unrestricted_parties = party_order
restricted_provinces = list(set(province_order) - set(unrestricted_provinces))
restricted_parties = list(set(unrestricted_parties) - set([restricted_party]))
# Enforce ordering
restricted_parties = [p for p in party_order if p in restricted_parties]
restricted_provinces = [p for p in province_order if p in restricted_provinces]
unrestricted_provinces = [p for p in province_order if p in unrestricted_provinces]


PLOTLY_TEMPLATE = 'simple_white'
logging.debug(f'SETTING PLOTLY_TEMPLATE TO {PLOTLY_TEMPLATE}')
