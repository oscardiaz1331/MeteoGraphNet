import logging
import os
from dotenv import load_dotenv
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
        load_dotenv()   
        self.params = {'api_key': os.environ.get("AEMET_API_KEY")}
        self.aemet_client = RestClient(
            base_url=BASE,
            default_headers={"accept": "application/json"},
            timeout=15.0,
        )
        self.data_client = RestClient(base_url="", timeout=15.0)

    def __del__(self):
        if hasattr(self, 'aemet_client'):
            self.aemet_client.close()
        if hasattr(self, 'data_client'):
            self.data_client.close()

    def get_municipios(self):
        resp = self.aemet_client.get("/opendata/api/maestro/municipios", params=self.params)
        payload = resp.json()
        data = self.data_client.get(payload['datos'])
        return data.json()