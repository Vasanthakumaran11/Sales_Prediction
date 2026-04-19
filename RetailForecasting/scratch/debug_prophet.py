import logging
import sys
from prophet import Prophet

# Set up logging to catch Prophet's initialization details
logger = logging.getLogger('prophet')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
logger.addHandler(handler)

try:
    print("Testing Prophet initialization...")
    m = Prophet()
    print("Prophet initialized successfully!")
except Exception as e:
    print(f"Prophet initialization failed: {type(e).__name__} - {e}")
    
import cmdstanpy
print(f"cmdstanpy version: {cmdstanpy.__version__}")
try:
    print(f"cmdstan path: {cmdstanpy.cmdstan_path()}")
except Exception as e:
    print(f"Could not find cmdstan path: {e}")
