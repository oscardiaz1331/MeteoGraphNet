import logging
import os
from RestClient import RestClientError, RestClient

class AEMETClient():
    """
    AEMET REST client that automatically includes the API key in requests.

    Parameters
    ----------
    base_url:
        The base URL / domain for the AEMET API (e.g. "https://opendata.aemet.es").
        It must not end with a trailing slash — the client will handle joining.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        BASE = "https://opendata.aemet.es"
        self.client = RestClient(
            base_url=BASE,
            auth_token=os.environ.get("AEMET_API_KEY"),
            default_headers={"accept": "application/json"},
            timeout=15.0,
        )

    def __del__(self):
        if hasattr(self, 'client'):
            self.client.close()

    def get_municipios(self):
        resp = self.client.get("/opendata/api/maestro/municipios")
        payload = resp.json()
        print(payload)  # this is the JSON containing "datos" and "metadatos"