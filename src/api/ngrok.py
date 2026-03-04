"""
ngrok tunnel management 
Controlled via NGROK_ENABLED, NGROK_AUTH_TOKEN, and NGROK_DOMAIN env vars.
"""
import logging
from pyngrok import ngrok, conf
from api.config import get_settings

logger = logging.getLogger("uvicorn.error")

_tunnel = None

def start_tunnel(port: int = 8000):
    """Open an ngrok tunnel if NGROK_ENABLED is True."""
    global _tunnel
    settings = get_settings()

    if not settings.ngrok_enabled:
        return

    if settings.ngrok_auth_token:
        conf.get_default().auth_token = settings.ngrok_auth_token

    # Kill any leftover ngrok processes before connecting
    ngrok.kill()

    kwargs = {"addr": str(port), "proto": "http"}
    if settings.ngrok_domain:
        kwargs["domain"] = settings.ngrok_domain

    try:
        _tunnel = ngrok.connect(**kwargs)
        logger.info(f"ngrok tunnel opened: {_tunnel.public_url}")
    except Exception as e:
        logger.warning(f"ngrok tunnel failed to start (API will continue without it): {e}")


def stop_tunnel():
    """Close the ngrok tunnel if one is active."""
    global _tunnel
    if _tunnel is not None:
        ngrok.disconnect(_tunnel.public_url)
        _tunnel = None
        logger.info("ngrok tunnel closed")