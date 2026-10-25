from prometheus_client import Counter, Histogram, Gauge
import time

#Define metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method','endpoint','status'] 
)

request_duration = Histogram(
    'api_request_duration_seconds'
    'API request duration'
    ['endpoint']
)

model_predictions = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['prediction','confidence_range']
)

def get_confidence_range(confidence):
    """Categorize confidence level"""
    if confidence < 0.5:
        return '0-50'
    elif confidence < 0.75:
        return '50-75'
    else:
        return '75-100'

class MetricsMiddleware:
    """Middleware for colecting metrics"""

    def __init__(self,app):
        self.app = app
    
    def __call__(self,environ,start_response):
        """Wrap WSGI app"""
        start_time = time.time()

        def custom_start_response(status,headers):
            return start_response(status,headers)
        
        result = self.app(environ,custom_start_response)

        duration = time.time() - start_time
        request.duration.labels(endpoint=environ.get('PATH_INFO')).observe(duration)

        return result
    